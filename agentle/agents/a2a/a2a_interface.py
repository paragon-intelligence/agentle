"""
A2A Interface

The Agent-to-Agent Interface that allows one agent to interact with another
by sending it tasks or subscribing to it.
"""

import asyncio
import logging
import threading
from typing import Optional, TYPE_CHECKING, TypeVar, Union

from agentle.agents.a2a.resources.push_notification_resource import (
    PushNotificationResource,
)
from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from agentle.agents.agent import Agent
    from agentle.agents.agent_pipeline import AgentPipeline
    from agentle.agents.agent_team import AgentTeam

logger = logging.getLogger(__name__)

# Define a type variable for the output schema
T_Schema = TypeVar("T_Schema")


def _get_event_loop():
    """Get the current event loop or create a new one."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run_async(coro):
    """Run a coroutine in the event loop safely."""
    loop = _get_event_loop()
    if loop.is_running():
        # We're in a context where the event loop is already running
        # For example, in a GUI application or web server
        return asyncio.ensure_future(coro)
    else:
        # We're in a synchronous context, so we can run the coroutine to completion
        return loop.run_until_complete(coro)


def _run_async_in_thread(coro, timeout=None):
    """
    Run a coroutine in a separate thread with its own event loop.

    This avoids issues with cancellations in nested event loops.
    """
    result_container = []
    exception_container = []

    def thread_target():
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
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Timeout occurred
        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    if exception_container:
        raise exception_container[0]

    return result_container[0]


class A2AInterface:
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
        agent: "Union[Agent[T_Schema], AgentTeam, AgentPipeline]",
        task_manager: Optional[TaskManager] = None,
    ):
        """
        Initialize the A2A interface.

        Args:
            agent: The agent to interact with
            task_manager: Optional task manager to use (default: InMemoryTaskManager)
        """
        if task_manager is None:
            # Import here to avoid circular import
            from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager

            task_manager = InMemoryTaskManager()

        # Create the task resource
        self.agent = agent
        self.task_manager = task_manager

        # Create enhanced task resource with safer execution
        self.tasks = TaskResource(agent=agent, manager=task_manager)

        # Override task methods with more reliable thread-based execution
        original_send = self.tasks.send
        original_get = self.tasks.get
        original_cancel = self.tasks.cancel

        # Enhance send method to use thread-based async execution
        def enhanced_send(task_params):
            try:
                return _run_async_in_thread(
                    self.task_manager.send(task_params=task_params, agent=self.agent),
                    timeout=30,
                )
            except Exception as e:
                logger.error(f"Error in enhanced send: {e}")
                raise

        # Enhance get method to use thread-based async execution
        def enhanced_get(query_params):
            try:
                return _run_async_in_thread(
                    self.task_manager.get(query_params=query_params, agent=self.agent),
                    timeout=10,
                )
            except Exception as e:
                logger.error(f"Error in enhanced get: {e}")
                raise

        # Enhance cancel method to use thread-based async execution
        def enhanced_cancel(task_id):
            try:
                return _run_async_in_thread(
                    self.task_manager.cancel(task_id=task_id), timeout=5
                )
            except Exception as e:
                logger.error(f"Error in enhanced cancel: {e}")
                raise

        # Apply the enhanced methods
        self.tasks.send = enhanced_send
        self.tasks.get = enhanced_get
        self.tasks.cancel = enhanced_cancel

        self.push_notifications = PushNotificationResource(agent=agent)
