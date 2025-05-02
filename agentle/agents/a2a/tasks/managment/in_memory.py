"""
A2A In-Memory Task Manager

This module provides an in-memory implementation of the TaskManager interface.
The InMemoryTaskManager stores tasks in a simple in-memory dictionary, which makes it
suitable for testing, development, and simple applications that don't require
persistent storage.
"""

import asyncio
import logging
from collections.abc import MutableSequence
from typing import Any, Dict

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

logger = logging.getLogger(__name__)


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
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_histories: Dict[str, MutableSequence[Message]] = {}
        self._message_adapter = GenerationMessageToMessageAdapter()
        self._a2a_to_generation_adapter = MessageToGenerationMessageAdapter()

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
        # Create a new Task object
        task = Task(
            id=task_params.id,
            sessionId=task_params.sessionId or task_params.id,
            status=TaskState.SUBMITTED,
            history=[task_params.message] if task_params.message else None,
        )

        # Store the task
        self._tasks[task.id] = task

        # Initialize history for the task
        history = self._task_histories.get(task.id, [])
        if not history and task_params.message:
            history.append(task_params.message)
        self._task_histories[task.id] = history

        # Create an asyncio task to run the agent
        asyncio_task = asyncio.create_task(
            self._run_agent_task(task.id, agent, task_params)
        )
        self._running_tasks[task.id] = asyncio_task

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
        if task_id not in self._tasks:
            return TaskGetResult(
                result=Task(
                    id=task_id,
                    sessionId=task_id,
                    status=TaskState.FAILED,
                ),
                error=f"Task with ID {task_id} not found",
            )

        task = self._tasks[task_id]

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
                id=task_params.id,
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
        if task_id not in self._tasks or task_id not in self._running_tasks:
            return False

        try:
            # Cancel the asyncio task
            asyncio_task = self._running_tasks[task_id]
            if not asyncio_task.done():
                asyncio_task.cancel()

                # Wait for the task to be canceled
                try:
                    await asyncio_task
                except asyncio.CancelledError:
                    pass  # Expected behavior when canceling

            # Update the task status
            task = self._tasks[task_id]
            task.status = TaskState.CANCELED

            # Clean up
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]

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
        try:
            # Update task status to WORKING
            task = self._tasks[task_id]
            task.status = TaskState.WORKING

            # Get the message history from the task
            history = self._task_histories.get(task_id, [])

            # Convert the A2A Message to a UserMessage for the agent
            gen_message = self._a2a_to_generation_adapter.adapt(task_params.message)
            # Only UserMessage is accepted, so ensure we have the right type
            if not isinstance(gen_message, UserMessage):
                gen_message = UserMessage(parts=gen_message.parts)

            # Run the agent
            result = await agent.run_async(gen_message)

            # Convert the assistant message to an A2A Message
            if result.generation.output:
                assistant_message = self._message_adapter.adapt(
                    result.generation.output
                )
                history.append(assistant_message)
                self._task_histories[task_id] = history

            # Update task with the result
            task.status = TaskState.COMPLETED

        except asyncio.CancelledError:
            # Task was canceled
            logger.info(f"Task {task_id} was canceled")
            task = self._tasks[task_id]
            task.status = TaskState.CANCELED

        except Exception as e:
            # Task failed
            logger.exception(f"Error executing task {task_id}: {e}")
            task = self._tasks[task_id]
            task.status = TaskState.FAILED

        finally:
            # Remove the running task reference
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
