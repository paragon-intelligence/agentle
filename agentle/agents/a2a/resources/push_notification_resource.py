"""
A2A Push Notification Resource

This module defines the PushNotificationResource class, which provides methods for configuring
and managing push notifications in the A2A protocol. It enables setting up notification
endpoints and retrieving notification configurations.
"""

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.notifications.push_notification_config import (
    PushNotificationConfig,
)
from agentle.agents.a2a.notifications.task_push_notification_config import (
    TaskPushNotificationConfig,
)
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.agent import Agent

type WithoutStructuredOutput = None


class PushNotificationResource[T_Schema = WithoutStructuredOutput](BaseModel):
    """
    Resource for configuring and managing push notifications.

    This class provides methods for setting up notification endpoints and retrieving
    notification configurations in the A2A protocol. It enables agents to send
    asynchronous updates to clients about task status and progress.

    Attributes:
        agent: The agent associated with the notifications

    Type Parameters:
        T_Schema: The structured output schema of the agent, defaults to None

    Example:
        ```python
        from agentle.agents.agent import Agent
        from agentle.agents.a2a.resources.push_notification_resource import PushNotificationResource
        from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

        # Create an agent
        agent = Agent(...)

        # Create a push notification resource
        notification_resource = PushNotificationResource(agent=agent)

        # Configure notifications
        notification_config = PushNotificationConfig(
            url="https://example.com/webhooks/notifications",
            token="notification-token-123"
        )

        task_notification = notification_resource.set(notification_config)

        # Retrieve notification configuration for a task
        query_params = {"id": task_notification.id}
        result = notification_resource.get(query_params)
        ```
    """

    agent: Agent[T_Schema]
    """The agent associated with the notifications"""

    def set(self, config: PushNotificationConfig) -> TaskPushNotificationConfig:
        """
        Sets up a push notification configuration.

        This method configures push notifications for tasks with the specified
        configuration. It returns a task notification configuration that links
        the notification settings to a specific task.

        Args:
            config: The push notification configuration to set up

        Returns:
            TaskPushNotificationConfig: The task notification configuration

        Example:
            ```python
            from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

            # Create a notification configuration
            notification_config = PushNotificationConfig(
                url="https://example.com/webhooks/notifications",
                token="notification-token-123"
            )

            # Set up the configuration
            task_notification = notification_resource.set(notification_config)
            print(f"Task ID: {task_notification.id}")
            ```
        """
        ...

    def get(self, query_params: TaskQueryParams) -> TaskGetResult:
        """
        Retrieves notification configuration for a task.

        This method retrieves the push notification configuration associated with
        a specific task, identified by the query parameters.

        Args:
            query_params: Parameters for querying the task

        Returns:
            TaskGetResult: The task result containing notification configuration

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams

            # Create query parameters
            query = TaskQueryParams(id="task-123")

            # Retrieve notification configuration
            result = notification_resource.get(query)

            if result.error:
                print(f"Error retrieving notification config: {result.error}")
            else:
                task = result.result
                print(f"Task ID: {task.id}")
                if task.metadata and "notification_url" in task.metadata:
                    print(f"Notification URL: {task.metadata['notification_url']}")
            ```
        """
        ...
