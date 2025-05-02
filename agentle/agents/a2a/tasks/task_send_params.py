"""
A2A Task Send Parameters

This module defines the TaskSendParams class, which represents the parameters for sending
a task to an agent in the A2A protocol. It encapsulates all the information needed to
create or continue a task.
"""

import uuid
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.notifications.push_notification_config import (
    PushNotificationConfig,
)


class TaskSendParams(BaseModel):
    """
    Parameters for sending a task to an agent in the A2A protocol.

    This class encapsulates all the information needed to create a new task or continue
    an existing conversation. It includes the message to send, session information,
    history configuration, and optional push notification settings.

    Attributes:
        id: Unique identifier for the task (auto-generated if not provided)
        sessionId: Optional identifier for the session holding the task
        message: The message to send to the agent
        historyLength: Optional limit on the number of history messages to include
        pushNotification: Optional configuration for push notifications
        metadata: Optional additional metadata associated with the task

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

        # Create a simple message
        message = Message(
            role="user",
            parts=[TextPart(text="What's the weather today?")]
        )

        # Create basic task parameters
        task_params = TaskSendParams(
            message=message,
            sessionId="weather-session"
        )

        # Create more advanced task parameters with notifications and history control
        notification_config = PushNotificationConfig(
            url="https://example.com/notifications",
            token="notification-token-123"
        )

        advanced_params = TaskSendParams(
            message=message,
            sessionId="weather-session",
            historyLength=5,  # Limit history to last 5 messages
            pushNotification=notification_config,
            metadata={"source": "weather_app", "user_location": "Seattle"}
        )
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """
    server creates a new sessionId for new tasks if not set
    """

    sessionId: str | None = Field(default=None)
    """
    client-generated id for the session holding the task.
    server creates a new sessionId for new tasks if not set
    """

    message: Message
    """
    The message to send to the agent
    """

    historyLength: int | None = Field(default=None)
    """
    Optional limit on the number of history messages to include
    """

    pushNotification: PushNotificationConfig | None = Field(default=None)
    """
    Optional configuration for push notifications
    """

    metadata: dict[str, Any] | None = Field(default=None)
    """
    Optional additional metadata associated with the task
    """
