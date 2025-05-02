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
import traceback
import uuid
from collections.abc import MutableSequence
from typing import Any, Dict, List, Optional

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
from agentle.agents.a2a.tasks.send_task_response import SendTaskResponse
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam
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


# Global event loop management
def get_or_create_eventloop():
    try:
        loop = asyncio.get_event_loop()
        logger.debug(f"Using existing event loop: {id(loop)}")
    except RuntimeError:
        # If we're not in the main thread, create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug(f"Created new event loop: {id(loop)}")
    return loop


def run_coroutine_sync(coroutine):
    """Run a coroutine synchronously, creating an event loop if needed."""
    loop = get_or_create_eventloop()
    logger.debug(
        f"Running coroutine in loop: {id(loop)}, is running: {loop.is_running()}"
    )
    if loop.is_running():
        # We're already in an event loop, so we can just run the coroutine
        future = asyncio.ensure_future(coroutine)
        logger.debug("Added coroutine to running loop")
        return future
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

    def __init__(self):
        """Initialize the in-memory task manager."""
        logger.debug("Initializing InMemoryTaskManager")
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_histories: Dict[str, MutableSequence[Message]] = {}
        self._message_adapter = GenerationMessageToMessageAdapter()
        self._a2a_to_generation_adapter = MessageToGenerationMessageAdapter()
        self._lock = threading.Lock()
        self._event_loop = get_or_create_eventloop()

    def _log_task_status(
        self, task_id: str, message: str, asyncio_task: asyncio.Task = None
    ):
        """Helper to log task status with detailed information."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task_status = task.status
            else:
                task_status = "NOT_FOUND"

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
        agent: Agent[Any] | AgentTeam | AgentPipeline,
    ) -> Task:
        """
        Creates and starts a new task or continues an existing session.

        This method creates a Task object, starts an asyncio task to run the
        agent, and stores both in the internal dictionaries.

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

        # Get the current event loop
        loop = get_or_create_eventloop()
        logger.debug(f"Using event loop {id(loop)} for task {task.id}")

        try:
            # Create an asyncio task to run the agent
            # This will run in the background while we immediately return the task
            logger.debug(f"Creating asyncio task for agent execution in task {task.id}")

            # Define a wrapper function that we can submit to the loop
            async def run_task_wrapper():
                try:
                    # Allow some time for event loop to stabilize
                    await asyncio.sleep(0.2)

                    # Double-check that the task hasn't been cancelled before we start
                    with self._lock:
                        if task_id not in self._tasks:
                            logger.debug(
                                f"Task {task_id} no longer exists, aborting wrapper"
                            )
                            return

                        task = self._tasks[task_id]
                        if task.status == TaskState.CANCELED:
                            logger.debug(
                                f"Task {task_id} already cancelled, aborting wrapper"
                            )
                            return

                    # Use a separate try block to catch errors during the actual task execution
                    try:
                        # Run the task in a way that's protected from cancellation
                        await self._run_agent_task(task.id, agent, task_params)
                    except asyncio.CancelledError:
                        logger.debug(
                            f"Task {task_id} was cancelled during execution phase"
                        )
                        with self._lock:
                            if task_id in self._tasks:
                                self._tasks[task_id].status = TaskState.CANCELED
                    except Exception as e:
                        logger.exception(f"Error in _run_agent_task for {task_id}: {e}")
                        with self._lock:
                            if task_id in self._tasks:
                                self._tasks[task_id].status = TaskState.FAILED
                except asyncio.CancelledError:
                    logger.debug(f"Task {task_id} was cancelled during wrapper setup")
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id].status = TaskState.CANCELED
                except Exception as e:
                    logger.exception(
                        f"Unhandled exception in task wrapper for {task_id}: {e}"
                    )
                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id].status = TaskState.FAILED

            # Start the task in the current event loop
            asyncio_task = None
            if loop.is_running():
                # Create a task in the running loop
                asyncio_task = asyncio.ensure_future(run_task_wrapper(), loop=loop)
                logger.debug(f"Created task in running loop for {task.id}")
            else:
                # Create a dedicated thread for running this task with its own event loop
                import threading

                def run_in_thread():
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        logger.debug(
                            f"Created dedicated thread loop for task {task.id}"
                        )
                        thread_loop.run_until_complete(run_task_wrapper())
                    except Exception as e:
                        logger.exception(f"Error in thread for task {task.id}: {e}")
                    finally:
                        thread_loop.close()

                # Start the thread
                thread = threading.Thread(target=run_in_thread)
                thread.daemon = True  # Allow program to exit if thread is still running
                thread.start()
                logger.debug(f"Started dedicated thread for task {task.id}")

                # Create a dummy task for tracking purposes
                asyncio_task = asyncio.ensure_future(asyncio.sleep(0))
                asyncio_task._thread = thread  # Attach thread reference to the task

            # Set up callback to log when the task is done
            if asyncio_task:
                asyncio_task.add_done_callback(
                    lambda t: logger.debug(
                        f"Asyncio task for {task.id} completed with state: {'Canceled' if t.cancelled() else 'Done'}"
                    )
                )

                with self._lock:
                    self._running_tasks[task.id] = asyncio_task
                    logger.debug(f"Registered async task for {task.id}")

        except Exception as e:
            logger.exception(f"Error creating asyncio task for {task.id}: {e}")
            with self._lock:
                # If we fail to create the asyncio task, mark the task as failed
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskState.FAILED

        self._log_task_status(task.id, "Task created")
        return task

    async def get(
        self,
        query_params: TaskQueryParams,
        agent: Agent[Any] | AgentTeam | AgentPipeline,
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
        agent: Agent[Any] | AgentTeam | AgentPipeline,
    ) -> JSONRPCResponse:
        """
        Sends a task and sets up a subscription for updates.

        Currently, this implementation just creates the task without setting up
        actual push notifications. It returns a JSON-RPC response indicating
        whether the task was created.

        Args:
            task_params: Parameters for the task to create
            agent: The agent to execute the task

        Returns:
            JSONRPCResponse: The response containing subscription information
        """
        try:
            task = await self.send(task_params, agent)
            return SendTaskResponse(id=task.id, result=task)
        except Exception as e:
            logger.exception("Error sending task with subscription")
            return SendTaskResponse(
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
        agent: Agent[Any] | AgentTeam | AgentPipeline,
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
        self._log_task_status(task_id, "Starting execution")

        # Safety check to ensure we can handle tasks and event loop properly
        try:
            # Print the current thread and event loop
            current_thread = threading.current_thread()
            logger.debug(
                f"Task {task_id} running in thread: {current_thread.name}, loop: {id(asyncio.get_event_loop())}"
            )
        except Exception as e:
            logger.exception(f"Error getting thread/loop info: {e}")

        try:
            # Update task status to WORKING
            with self._lock:
                if task_id not in self._tasks:
                    logger.warning(
                        f"Task {task_id} no longer exists, aborting execution"
                    )
                    return
                task = self._tasks[task_id]
                prev_status = task.status
                task.status = TaskState.WORKING
                logger.debug(
                    f"Updated task {task_id} status from {prev_status} to {task.status}"
                )

            # Get the message history from the task
            history = self._task_histories.get(task_id, [])

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

            # Double-check task still exists (wasn't canceled during execution)
            with self._lock:
                if task_id not in self._tasks:
                    logger.warning(f"Task {task_id} was removed during execution")
                    return
                task = self._tasks[task_id]
                # If task was explicitly canceled while we were processing, don't change its status
                if task.status == TaskState.CANCELED:
                    logger.debug(
                        f"Task {task_id} was canceled during execution, preserving status"
                    )
                    return

            # Process the agent's response
            logger.debug(f"Processing agent response for task {task_id}")

            # Access the message from the generation result
            # The Generation class has 'choices', and the first choice contains the message
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
                    assistant_message = self._message_adapter.adapt(output_message)
                    history.append(assistant_message)
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

                assistant_message = Message(
                    role="agent", parts=[TextPart(text=default_text)]
                )
                history.append(assistant_message)
                with self._lock:
                    self._task_histories[task_id] = history
                    logger.debug(f"Task {task_id} - Added default message to history")

            # Update task with the result - use lock to prevent race conditions
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    # Only update status if it's still WORKING (not manually CANCELED)
                    if task.status == TaskState.WORKING:
                        prev_status = task.status
                        task.status = TaskState.COMPLETED
                        logger.debug(
                            f"Updated task {task_id} status from {prev_status} to {task.status}"
                        )
                    else:
                        logger.debug(
                            f"Not updating task {task_id} status as it's already {task.status}"
                        )

        except asyncio.CancelledError:
            # Task was explicitly canceled
            logger.info(f"Task {task_id} was explicitly canceled")
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    prev_status = task.status
                    task.status = TaskState.CANCELED
                    logger.debug(
                        f"Updated task {task_id} status from {prev_status} to {task.status} due to cancellation"
                    )

        except Exception as e:
            # Task failed
            logger.exception(f"Error executing task {task_id}: {e}")

            # Print full traceback for debugging
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
            logger.debug(
                f"Task {task_id} failure traceback: {''.join(traceback_details)}"
            )

            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    prev_status = task.status
                    task.status = TaskState.FAILED
                    logger.debug(
                        f"Updated task {task_id} status from {prev_status} to {task.status} due to failure"
                    )

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
            List[Task]: List of tasks matching the query
        """
        with self._lock:
            # Return a copy of all tasks if no query params provided
            if not query_params:
                return list(self._tasks.values())

            # Filter tasks by session ID if provided
            if query_params.sessionId:
                return [
                    task
                    for task in self._tasks.values()
                    if task.sessionId == query_params.sessionId
                ]

            # Return a specific task by ID if provided
            if query_params.id and query_params.id in self._tasks:
                return [self._tasks[query_params.id]]

            # No matching tasks
            return []
