"""
A2A Task Resource

This module defines the TaskResource class, which provides methods for interacting with tasks
in the A2A protocol. The TaskResource acts as an interface for sending tasks to agents,
retrieving task results, and managing task notifications.
"""

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.resources.push_notification_resource import (
    PushNotificationResource,
)
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam

type WithoutStructuredOutput = None


class TaskResource[T_Schema = WithoutStructuredOutput](BaseModel):
    """
    Provides methods for interacting with tasks in the A2A protocol.

    The TaskResource acts as an interface for sending tasks to agents, retrieving task
    results, and managing task notifications. It works with individual agents, agent teams,
    and agent pipelines.

    Attributes:
        agent: The agent, agent team, or agent pipeline to interact with
        manager: The task manager responsible for handling tasks

    Example:
        ```python
        from agentle.agents.agent import Agent
        from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
        from agentle.agents.a2a.resources.task_resource import TaskResource
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart

        # Create an agent and task manager
        agent = Agent(...)
        task_manager = TaskManager()

        # Create the task resource
        task_resource = TaskResource(agent=agent, manager=task_manager)

        # Prepare a task
        message = Message(
            role="user",
            parts=[TextPart(text="What is the capital of France?")]
        )
        task_params = TaskSendParams(
            message=message,
            sessionId="geography-session"
        )

        # Send the task and get results
        task = task_resource.send(task_params)
        result = task_resource.get(query_params={"id": task.id})
        ```
    """

    agent: Agent[T_Schema] | AgentTeam[T_Schema] | AgentPipeline[T_Schema]
    """The agent, agent team, or agent pipeline to interact with"""

    manager: TaskManager
    """The task manager responsible for handling tasks"""

    @property
    def pushNotification(self) -> PushNotificationResource[T_Schema]:
        """
        Access to push notification configuration.

        This property provides access to the PushNotificationResource, which allows
        for setting up and managing push notifications for task updates.

        Returns:
            PushNotificationResource: The push notification resource

        Example:
            ```python
            from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

            # Configure push notifications
            config = PushNotificationConfig(
                url="https://example.com/notifications",
                token="notification-token-123"
            )
            notification_config = task_resource.pushNotification.set(config)
            ```
        """
        return PushNotificationResource(agent=self.agent)

    def send(self, task: TaskSendParams) -> Task:
        """
        Sends a task to the agent.

        This method sends a task with the specified parameters to the agent for processing.
        It creates a new task or continues an existing session based on the provided parameters.

        Args:
            task: The parameters for the task to send

        Returns:
            Task: The created or updated task

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
            from agentle.agents.a2a.messages.message import Message
            from agentle.agents.a2a.message_parts.text_part import TextPart

            # Create a message and task parameters
            message = Message(
                role="user",
                parts=[TextPart(text="Translate 'hello' to French")]
            )
            task_params = TaskSendParams(
                message=message,
                sessionId="translation-session"
            )

            # Send the task
            task = task_resource.send(task_params)
            print(f"Task created with ID: {task.id}")
            ```
        """
        return self.manager.send(task, agent=self.agent)

    def get(self, query_params: TaskQueryParams) -> TaskGetResult:
        """
        Retrieves a task result.

        This method retrieves the result of a task based on the specified query parameters.
        It is typically used to get the response after sending a task.

        Args:
            query_params: The parameters for querying the task

        Returns:
            TaskGetResult: The result of the task

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams

            # Get results for a specific task
            query = TaskQueryParams(id="task-123")
            result = task_resource.get(query)

            # Access the agent's response
            if result.result.history and len(result.result.history) > 1:
                agent_response = result.result.history[1]
                for part in agent_response.parts:
                    if part.type == "text":
                        print(part.text)
            ```
        """
        return self.manager.get(query_params, agent=self.agent)

    def send_subscribe(self, task: TaskSendParams) -> JSONRPCResponse:
        """
        Sends a task and subscribes to updates.

        This method sends a task to the agent and sets up a subscription to receive
        updates about the task's progress. It is useful for long-running tasks where
        you want to be notified of state changes.

        Args:
            task: The parameters for the task to send

        Returns:
            JSONRPCResponse: The response containing subscription information

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
            from agentle.agents.a2a.messages.message import Message
            from agentle.agents.a2a.message_parts.text_part import TextPart
            from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

            # Create a message, notification config, and task parameters
            message = Message(
                role="user",
                parts=[TextPart(text="Analyze this dataset")]
            )

            notification_config = PushNotificationConfig(
                url="https://example.com/notifications",
                token="notification-token-123"
            )

            task_params = TaskSendParams(
                message=message,
                sessionId="analysis-session",
                pushNotification=notification_config
            )

            # Send the task and subscribe to updates
            response = task_resource.send_subscribe(task_params)
            print(f"Subscription ID: {response.id}")
            ```
        """
        return self.manager.send_subscribe(task, agent=self.agent)
