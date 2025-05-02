"""
A2A Task Push Notification Configuration

This module defines the TaskPushNotificationConfig class, which associates a task ID
with push notification configuration in the A2A protocol. It enables tracking which
notifications are associated with specific tasks.
"""

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.notifications.push_notification_config import (
    PushNotificationConfig,
)


class TaskPushNotificationConfig(BaseModel):
    """
    Associates a task ID with push notification configuration.

    This class links a task identifier with its corresponding push notification
    configuration, enabling tracking of which notifications are associated with
    specific tasks in the A2A protocol.

    Attributes:
        id: The identifier of the task
        pushNotificationConfig: The push notification configuration for the task

    Example:
        ```python
        from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig
        from agentle.agents.a2a.notifications.task_push_notification_config import TaskPushNotificationConfig

        # Create a push notification configuration
        notification_config = PushNotificationConfig(
            url="https://example.com/webhooks/notifications",
            token="notification-token-123"
        )

        # Associate it with a task
        task_notification = TaskPushNotificationConfig(
            id="task-456",
            pushNotificationConfig=notification_config
        )

        # Access the task ID and notification URL
        print(f"Task ID: {task_notification.id}")
        print(f"Notification URL: {task_notification.pushNotificationConfig.url}")
        ```

    Note:
        This class is typically used as a return value when setting up notifications
        for a task, confirming the configuration has been applied to the specified task.
    """

    id: str
    """The identifier of the task"""

    pushNotificationConfig: PushNotificationConfig
    """The push notification configuration for the task"""
