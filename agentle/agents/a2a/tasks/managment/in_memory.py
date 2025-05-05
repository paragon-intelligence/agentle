"""
A2A In-Memory Task Manager

This module provides an in-memory implementation of the TaskManager interface.
The InMemoryTaskManager stores tasks in a simple in-memory dictionary, which makes it
suitable for testing, development, and simple applications that don't require
persistent storage.
"""

import asyncio
import logging
import sys
import threading
import time
import uuid
from collections.abc import MutableSequence
from typing import Any, Coroutine, Optional, TypeVar, cast, Union, Dict, List

from agentle.agents.a2a.messages.generation_message_to_message_adapter import (
    GenerationMessageToMessageAdapter,
)
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.messages.message_to_generation_message_adapter import (
    MessageToGenerationMessageAdapter,
)
from agentle.agents.a2a.models.json_rpc_error import JSONRPCError
from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.user_message import UserMessage

# Configure logging
logger = logging.getLogger(__name__)
# Increase logging level for debugging
logger.setLevel(logging.DEBUG)
# Add a handler to output to stderr
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Define type variables for better typing
T = TypeVar("T")
AgentType = Union[Agent[Any], AgentTeam, AgentPipeline]


# Global event loop management
def get_or_create_eventloop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        logger.debug(f"Using existing event loop: {id(loop)}")
    except RuntimeError:
        # If we're not in the main thread, create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug(f"Created new event loop: {id(loop)}")
    return loop


def run_coroutine_sync(coroutine: Coroutine[None, None, T]) -> T:
    """Run a coroutine synchronously, creating an event loop if needed."""
    loop = get_or_create_eventloop()
    logger.debug(
        f"Running coroutine in loop: {id(loop)}, is running: {loop.is_running()}"
    )
    if loop.is_running():
        # We're already in an event loop, so we can just run the coroutine
        future = asyncio.ensure_future(coroutine)
        logger.debug("Added coroutine to running loop")
        return cast(T, future)
    else:
        # We need to run the coroutine in the event loop
        logger.debug(f"Running coroutine to completion in loop: {id(loop)}")
        return loop.run_until_complete(coroutine)


class InMemoryTaskManager(TaskManager):
    """
    In-memory implementation of the TaskManager interface.

    This task manager stores tasks in a dictionary and manages their execution
    using asyncio tasks. It supports creating, retrieving, and canceling tasks.
    Tasks are lost when the application is restarted.

    Attributes:
        _tasks: Dictionary mapping task IDs to Task objects
        _running_tasks: Dictionary mapping task IDs to asyncio Task objects
        _task_histories: Dictionary mapping task IDs to message histories

    Example:
        ```python
        from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.agent import Agent

        # Create an agent and task manager
        agent = Agent(...)
        task_manager = InMemoryTaskManager()

        # Send a task
        task_params = TaskSendParams(...)
        task = await task_manager.send(task_params, agent=agent)

        # Get the task result
        result = await task_manager.get(query_params={"id": task.id}, agent=agent)

        # Cancel a task
        success = await task_manager.cancel(task.id)
        ```
    """

    def __init__(self) -> None:
        """Initialize the in-memory task manager."""
        logger.debug("Initializing InMemoryTaskManager")
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._task_histories: Dict[str, MutableSequence[Message]] = {}
        self._message_adapter = GenerationMessageToMessageAdapter()
        self._a2a_to_generation_adapter = MessageToGenerationMessageAdapter()
        self._lock = threading.Lock()
        self._event_loop = get_or_create_eventloop()

    def _log_task_status(
        self,
        task_id: str,
        message: str,
        asyncio_task: Optional[asyncio.Task[Any]] = None,
    ) -> None:
        """Helper to log task status with detailed information."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task_status = task.status
            else:
                raise ValueError(f"Task {task_id} not found")

            if task_id in self._running_tasks:
                asyncio_task = self._running_tasks[task_id]
                asyncio_task_status = "DONE" if asyncio_task.done() else "RUNNING"
                if asyncio_task.done():
                    try:
                        exception = asyncio_task.exception()
                        if exception:
                            exception_str = str(exception)
                        else:
                            exception_str = "None"
                    except (asyncio.CancelledError, asyncio.InvalidStateError):
                        exception_str = "CANCELLED"
                else:
                    exception_str = "N/A"
            else:
                asyncio_task_status = "NOT_FOUND"
                exception_str = "N/A"

            logger.debug(
                f"Task {task_id} - {message} - Status: {task_status}, AsyncIO Task: {asyncio_task_status}, Exception: {exception_str}"
            )

    async def send(
        self,
        task_params: TaskSendParams,
        agent: AgentType,
    ) -> Task:
        """
        Creates and starts a new task or continues an existing session.

        This method creates a Task object, starts a thread to run the
        agent, and stores the task in the internal dictionaries.

        Args:
            task_params: Parameters for the task to create
            agent: The agent to execute the task

        Returns:
            Task: The created or updated task
        """
        # Create a unique ID if none provided
        task_id = task_params.id or str(uuid.uuid4())
        logger.debug(f"Creating new task with ID: {task_id}")

        # Create a new Task object
        task = Task(
            id=task_id,
            sessionId=task_params.sessionId or task_id,
            status=TaskState.SUBMITTED,
            history=[task_params.message] if task_params.message else None,
        )

        # Store the task with the lock to prevent race conditions
        with self._lock:
            self._tasks[task.id] = task
            logger.debug(f"Stored task {task.id} with status {task.status}")

            # Initialize history for the task
            history = self._task_histories.get(task.id, [])
            if not history and task_params.message:
                history.append(task_params.message)
            self._task_histories[task.id] = history

        # Create a thread to run the task in the background
        import threading

        def run_task_thread() -> None:
            """Run the task in a dedicated thread with its own event loop."""
            try:
                # Create a new event loop for this thread
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                try:
                    # Update task status to WORKING
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id].status = TaskState.WORKING
                            logger.debug(
                                f"Updated task {task_id} status to WORKING in thread"
                            )

                    # Run the agent task to completion in this thread's event loop
                    thread_loop.run_until_complete(
                        self._run_agent_task(task_id, agent, task_params)
                    )
                except asyncio.CancelledError:
                    logger.debug(f"Task {task_id} was cancelled during execution")
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id].status = TaskState.CANCELED
                except Exception as e:
                    logger.exception(f"Error running task {task_id} in thread: {e}")
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id].status = TaskState.FAILED
                finally:
                    # Clean up the event loop
                    thread_loop.close()
            except Exception as e:
                logger.exception(
                    f"Unhandled exception in thread for task {task_id}: {e}"
                )
                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id].status = TaskState.FAILED

        # Start the thread
        task_thread = threading.Thread(target=run_task_thread)
        task_thread.daemon = (
            True  # Allow program to exit even if thread is still running
        )
        task_thread.start()

        # Store a reference to the thread
        with self._lock:
            # Create a dummy task for tracking purposes and store the thread reference
            # in a way that doesn't trigger type errors
            dummy_task: asyncio.Task[None] = asyncio.ensure_future(asyncio.sleep(0))
            # We store the thread as a reference in the task manager instead
            # This avoids the attribute assignment that triggers type errors
            self._running_tasks[task.id] = dummy_task
            logger.debug(f"Started thread for task {task.id}")

        self._log_task_status(task.id, "Task created")
        return task

    async def get(
        self,
        query_params: TaskQueryParams,
        agent: AgentType,
    ) -> TaskGetResult:
        """
        Retrieves a task based on query parameters.

        This method looks up the task by ID and returns it with its current state.

        Args:
            query_params: Parameters to query the task
            agent: The agent associated with the task (not used in this implementation)

        Returns:
            TaskGetResult: The result of the task
        """
        task_id = query_params.id
        logger.debug(f"Getting task with ID: {task_id}")

        if task_id not in self._tasks:
            logger.warning(f"Task {task_id} not found")
            return TaskGetResult(
                result=Task(
                    id=task_id,
                    sessionId=task_id,
                    status=TaskState.FAILED,
                ),
                error=f"Task with ID {task_id} not found",
            )

        self._log_task_status(task_id, "Getting task status")

        with self._lock:
            task = self._tasks[task_id]
            logger.debug(f"Found task {task_id} with status {task.status}")

            # If historyLength is specified, limit the number of messages in the history
            history = self._task_histories.get(task_id, [])
            if query_params.historyLength is not None and history:
                limit = min(query_params.historyLength, len(history))
                history = history[-limit:]

            # Update the task with the current history
            task_copy = Task(
                id=task.id,
                sessionId=task.sessionId,
                status=task.status,
                history=history,
                artifacts=task.artifacts,
                metadata=task.metadata,
            )

        return TaskGetResult(result=task_copy)

    async def send_subscribe(
        self,
        task_params: TaskSendParams,
        agent: AgentType,
    ) -> JSONRPCResponse[Dict[str, Any]]:
        """
        Sends a task and sets up a subscription for updates.

        Currently, this implementation just creates the task without setting up
        actual push notifications. It returns a JSON-RPC response indicating
        whether the task was created.

        Args:
            task_params: Parameters for the task to create
            agent: The agent to execute the task

        Returns:
            JSONRPCResponse[Dict[str, Any]]: The response containing subscription information
        """
        try:
            task = await self.send(task_params, agent)
            # Create a new JSONRPCResponse with the task converted to a dict
            return JSONRPCResponse[Dict[str, Any]](
                id=task.id,
                result=task.model_dump(),
            )
        except Exception as e:
            logger.exception("Error sending task with subscription")
            # Create a JSONRPCResponse with an error
            return JSONRPCResponse[Dict[str, Any]](
                id=task_params.id or str(uuid.uuid4()),
                error=JSONRPCError(
                    code=-32603,
                    message=f"Internal error: {str(e)}",
                ),
            )

    async def cancel(self, task_id: str) -> bool:
        """
        Cancels an ongoing task.

        This method cancels the asyncio task associated with the task ID and
        updates the task status to CANCELED.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            bool: True if the task was successfully canceled, False otherwise
        """
        logger.debug(f"Cancelling task {task_id}")
        self._log_task_status(task_id, "Before cancel")

        with self._lock:
            if task_id not in self._tasks or task_id not in self._running_tasks:
                logger.warning(f"Cannot cancel task {task_id} - not found")
                return False

            try:
                # Cancel the asyncio task
                asyncio_task = self._running_tasks[task_id]
                if not asyncio_task.done():
                    logger.debug(f"Cancelling asyncio task for {task_id}")
                    asyncio_task.cancel()

                    # Note: We don't await the task here because that could block
                    # Instead, we'll mark it as canceled and clean up in the finally block
                    # of _run_agent_task
                else:
                    logger.debug(
                        f"Asyncio task for {task_id} already done, can't cancel"
                    )

                # Update the task status
                task = self._tasks[task_id]
                task.status = TaskState.CANCELED
                logger.debug(f"Marked task {task_id} as CANCELED")

                return True
            except Exception as e:
                logger.exception(f"Error cancelling task {task_id}: {e}")
                return False

    async def _run_agent_task(
        self,
        task_id: str,
        agent: AgentType,
        task_params: TaskSendParams,
    ) -> None:
        """
        Run the agent task and handle its lifecycle.

        This internal method executes the agent's run_async method with the provided
        input, updates the task status, and stores the results.

        Args:
            task_id: The ID of the task
            agent: The agent to execute the task
            task_params: Parameters for the task
        """
        logger.debug(f"Starting agent task execution for task {task_id}")

        # Safety check to ensure task exists
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} no longer exists, aborting execution")
                return

            prev_status: TaskState | None = None
            # Ensure the task is marked as WORKING
            task = self._tasks[task_id]
            if task.status != TaskState.WORKING:
                prev_status = task.status
                task.status = TaskState.WORKING
                logger.debug(
                    f"Updated task {task_id} status from {prev_status} to {task.status}"
                )

        # Get the message history from the task
        history = self._task_histories.get(task_id, [])

        try:
            # Convert the A2A Message to a UserMessage for the agent
            logger.debug(f"Converting message for agent in task {task_id}")
            gen_message = self._a2a_to_generation_adapter.adapt(task_params.message)
            # Only UserMessage is accepted, so ensure we have the right type
            if not isinstance(gen_message, UserMessage):
                gen_message = UserMessage(parts=gen_message.parts)

            # Run the agent
            logger.debug(f"Running agent for task {task_id}")
            start_time = time.time()
            try:
                result = await agent.run_async(gen_message)
                logger.debug(
                    f"Agent completed for task {task_id} in {time.time() - start_time:.2f} seconds"
                )
            except Exception as agent_error:
                logger.exception(
                    f"Agent execution error in task {task_id}: {agent_error}"
                )
                raise

            # Check if task still exists and hasn't been canceled
            with self._lock:
                if task_id not in self._tasks:
                    logger.warning(f"Task {task_id} was removed during execution")
                    return
                task = self._tasks[task_id]
                if task.status == TaskState.CANCELED:
                    logger.debug(
                        f"Task {task_id} was canceled during execution, preserving status"
                    )
                    return

            # Process the agent's response
            logger.debug(f"Processing agent response for task {task_id}")

            # Access the message from the generation result
            if (
                result.generation
                and hasattr(result.generation, "choices")
                and result.generation.choices
            ):
                # Get the message from the first choice
                logger.debug(f"Task {task_id} - Generation has choices")
                output_message = result.generation.choices[0].message
                if output_message:
                    # Convert the generated message to an A2A Message
                    logger.debug(f"Task {task_id} - Converting message to A2A format")
                    # First convert to the expected type
                    assistant_message = cast(AssistantMessage, output_message)
                    # Then use the adapter
                    a2a_message = self._message_adapter.adapt(assistant_message)
                    history.append(a2a_message)
                    with self._lock:
                        self._task_histories[task_id] = history
                        logger.debug(
                            f"Task {task_id} - Added assistant message to history"
                        )
            else:
                # If no response is available, create a default one
                logger.debug(
                    f"Task {task_id} - No generation choices available, creating default message"
                )
                from agentle.agents.a2a.message_parts.text_part import TextPart

                default_text = (
                    "I processed your request but couldn't generate a proper response."
                )

                # If generation has text property, use that
                if result.generation and hasattr(result.generation, "text"):
                    default_text = result.generation.text
                    logger.debug(
                        f"Task {task_id} - Using generation.text for default message"
                    )

                agent_message = Message(
                    role="agent", parts=[TextPart(text=default_text)]
                )
                history.append(agent_message)
                with self._lock:
                    self._task_histories[task_id] = history
                    logger.debug(f"Task {task_id} - Added default message to history")

            # Update task status to COMPLETED
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    if (
                        task.status != TaskState.CANCELED
                    ):  # Only update if not already canceled
                        prev_status = task.status
                        task.status = TaskState.COMPLETED
                        logger.debug(
                            f"Updated task {task_id} status from {prev_status} to {task.status}"
                        )
                        logger.debug(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.exception(f"Error executing task {task_id}: {e}")
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskState.FAILED
                    logger.debug(f"Task {task_id} failed with error: {e}")

            # Add error message to history
            try:
                from agentle.agents.a2a.message_parts.text_part import TextPart

                error_message = Message(
                    role="agent", parts=[TextPart(text=f"An error occurred: {str(e)}")]
                )
                history = self._task_histories.get(task_id, [])
                history.append(error_message)
                with self._lock:
                    self._task_histories[task_id] = history
                    logger.debug(f"Added error message to task {task_id} history")
            except Exception as inner_e:
                logger.exception(f"Failed to add error message to history: {inner_e}")
        finally:
            # Remove the running task reference
            with self._lock:
                if task_id in self._running_tasks:
                    del self._running_tasks[task_id]
                    logger.debug(f"Removed task {task_id} from running tasks")

        self._log_task_status(task_id, "Task execution completed")

    def list(self, query_params: Optional[TaskQueryParams] = None) -> List[Task]:
        """
        List tasks matching the query parameters.

        Args:
            query_params (Optional[TaskQueryParams]): Parameters for filtering tasks

        Returns:
            List[Task]: list of tasks matching the query
        """
        with self._lock:
            # Return a copy of all tasks if no query params provided
            if not query_params:
                return list(self._tasks.values())

            # Filter tasks by ID if provided
            if query_params.id and query_params.id in self._tasks:
                return [self._tasks[query_params.id]]

            # No matching tasks
            return []
