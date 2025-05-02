"""
A2A Interface

The Agent-to-Agent Interface that allows one agent to interact with another
by sending it tasks or subscribing to it.
"""

import logging
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
        self.tasks = TaskResource(agent=agent, manager=task_manager)
        self.push_notifications = PushNotificationResource(agent=agent)
