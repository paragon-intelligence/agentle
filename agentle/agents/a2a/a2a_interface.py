"""
A2A Interface

The Agent-to-Agent Interface that allows one agent to interact with another
by sending it tasks or subscribing to it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    cast,
)

from agentle.agents.a2a.a2a_interface_options import A2AInterfaceOptions
from agentle.agents.a2a.resources.push_notification_resource import (
    PushNotificationResource,
)
from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from agentle.agents.agent import Agent
    from agentle.agents.agent_pipeline import AgentPipeline
    from agentle.agents.agent_team import AgentTeam

logger = logging.getLogger(__name__)


def _run_async_in_thread[R](
    coro: Coroutine[Any, Any, R], timeout: Optional[float] = None
) -> R:
    """
    Run a coroutine in a separate thread with its own event loop.

    This avoids issues with cancellations in nested event loops.
    """
    result_container: list[Any] = []
    exception_container: list[Exception] = []

    def thread_target() -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
            finally:
                loop.close()
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join(timeout=timeout)  # Can be None for infinite waiting

    if thread.is_alive() and timeout is not None:
        # Only raise a timeout error if a timeout was specified
        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    if exception_container:
        raise exception_container[0]

    return cast(R, result_container[0])


# Create a wrapper class for task operations
class TasksWrapper[T_Schema]:
    """
    Wrapper for task operations with thread-safe async/sync conversion.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        agent: "Agent[T_Schema] | AgentTeam | AgentPipeline",
    ) -> None:
        """Initialize with the task manager and agent."""
        self.task_manager = task_manager
        self.agent = agent

    def send(self, task_params: TaskSendParams) -> Task:
        """Send a task to the agent."""
        try:
            return _run_async_in_thread(
                self.task_manager.send(task_params=task_params, agent=self.agent),
                timeout=None,
            )
        except Exception as e:
            logger.error(f"Error in send: {e}")
            raise

    def get(self, query_params: TaskQueryParams) -> TaskGetResult:
        """Get a task result."""
        try:
            return _run_async_in_thread(
                self.task_manager.get(query_params=query_params, agent=self.agent),
                timeout=None,
            )
        except Exception as e:
            logger.error(f"Error in get: {e}")
            raise

    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            return _run_async_in_thread(
                self.task_manager.cancel(task_id=task_id),
                timeout=None,
            )
        except Exception as e:
            logger.error(f"Error in cancel: {e}")
            raise


class A2AInterface[T_Schema]:
    """
    Agent-to-Agent Interface

    This class provides an interface for one agent to interact with another
    by sending it tasks or subscribing to it.

    Attributes:
        tasks: Resource for creating, retrieving, and canceling tasks
        push_notifications: Resource for subscribing to push notifications

    Example:
        ```python
        from agentle.agents.a2a.a2a_interface import A2AInterface
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.agent import Agent

        # Create an agent
        agent = Agent(...)

        # Create an A2A interface
        a2a = A2AInterface(agent=agent)

        # Create a message
        message = Message(
            role="user",
            parts=[TextPart(text="What is the meaning of life?")],
        )

        # Send a task to the agent
        task_params = TaskSendParams(
            message=message,
            sessionId="session-1",
        )
        task = a2a.tasks.send(task_params)

        # Get the result of the task
        task_result = a2a.tasks.get({"id": task.id})
        ```
    """

    def __init__(
        self,
        agent: "Agent[T_Schema] | AgentTeam | AgentPipeline",
        *,
        task_manager: Optional[TaskManager] = None,
        options: A2AInterfaceOptions | None = None,
    ):
        """
        Initialize the A2A interface.

        Args:
            agent: The agent to interact with
            task_manager: Optional task manager to use (default: InMemoryTaskManager)
        """
        from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager

        task_manager = task_manager or InMemoryTaskManager()

        # Create the task resource
        self.agent = agent
        self.task_manager = task_manager
        self.options = options

        # Use a wrapper for tasks instead of modifying the Pydantic model
        self.tasks = TasksWrapper[T_Schema](task_manager, agent)

        # Keep the original TaskResource for compatibility if needed
        self._task_resource = TaskResource(agent=agent, manager=task_manager)

        # Push notifications
        self.push_notifications = PushNotificationResource(agent=agent)
