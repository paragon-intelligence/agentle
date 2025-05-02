"""
A2A Notifications Package

This package provides classes for handling push notifications in the A2A protocol.
Push notifications allow agents to asynchronously notify clients about task status
changes and updates.

The package includes:
- Models for notification configuration
- Utilities for managing notification subscriptions
- Interfaces for handling notification delivery

Example:
    ```python
    from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig
    from agentle.agents.a2a.tasks.task_send_params import TaskSendParams

    # Create a push notification configuration
    notification_config = PushNotificationConfig(
        url="https://example.com/webhooks/notifications",
        token="notification-token-123"
    )

    # Use it in task parameters
    task_params = TaskSendParams(
        message=message,
        sessionId="task-session",
        pushNotification=notification_config
    )
    ```
"""

from agentle.agents.a2a.notifications.push_notification_config import (
    PushNotificationConfig,
)
from agentle.agents.a2a.notifications.task_push_notification_config import (
    TaskPushNotificationConfig,
)

__all__ = ["PushNotificationConfig", "TaskPushNotificationConfig"]
