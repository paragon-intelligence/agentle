"""
A2A Interface Implementation

This module provides the primary interface for interacting with agents using the A2A protocol.
The A2AInterface class serves as the main entry point for accessing agent capabilities through
a standardized task-based interface.

The A2A interface supports interaction with individual agents, agent teams, and agent pipelines,
providing a consistent protocol regardless of the underlying agent implementation.
"""

from typing import Any
from rsb.models.base_model import BaseModel

from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam


class A2AInterface[T_Schema = Any](BaseModel):
    """
    Main interface for interacting with agents using the A2A protocol.

    This class provides a standardized way to interact with different types of agents
    (individual agents, agent teams, or agent pipelines) through a task-based interface.

    Attributes:
        agent: The agent, agent team, or agent pipeline to interact with
        task_manager: Manager responsible for handling tasks and their lifecycle

    Example:
        ```python
        from agentle.agents.agent import Agent
        from agentle.agents.a2a.a2a_interface import A2AInterface
        from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart

        # Set up agent and interface
        agent = Agent(...)
        task_manager = TaskManager()
        a2a = A2AInterface(agent=agent, task_manager=task_manager)

        # Create a message and task parameters
        message = Message(
            role="user",
            parts=[TextPart(text="What is machine learning?")]
        )
        task_params = TaskSendParams(
            message=message,
            sessionId="session-123"
        )

        # Send a task and get the result
        task = a2a.tasks.send(task_params)
        result = a2a.tasks.get(query_params={"id": task.id})
        ```
    """

    agent: Agent[T_Schema] | AgentTeam | AgentPipeline
    """The agent, agent team, or agent pipeline to interact with"""

    task_manager: TaskManager
    """Manager responsible for handling tasks and their lifecycle"""

    @property
    def tasks(self) -> TaskResource[T_Schema]:
        """
        Access to task-related operations.

        This property provides access to the TaskResource, which allows for sending tasks to
        the agent, retrieving task results, and managing task notifications.

        Returns:
            TaskResource: The task resource for interacting with the agent

        Example:
            ```python
            # Send a task
            task = a2a.tasks.send(task_params)

            # Get task results
            result = a2a.tasks.get(query_params={"id": task.id})

            # Set up push notifications
            a2a.tasks.pushNotification.set(notification_config)
            ```
        """
        return TaskResource(agent=self.agent, manager=self.task_manager)
