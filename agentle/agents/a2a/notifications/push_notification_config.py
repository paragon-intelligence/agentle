"""
A2A Push Notification Configuration

This module defines the PushNotificationConfig class, which represents the configuration
for push notifications in the A2A protocol. It specifies how and where notifications should
be sent for asynchronous updates about task status and progress.
"""

# interface PushNotificationConfig {
#   url: string;
#   token?: string; // token unique to this task/session
#   authentication?: {
#     schemes: string[];
#     credentials?: string;
#   };
# }

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.models.authentication import Authentication


class PushNotificationConfig(BaseModel):
    """
    Configuration for push notifications in the A2A protocol.

    This class specifies how and where notifications should be sent for asynchronous
    updates about task status and progress. It includes the webhook URL, an optional
    token for identification, and optional authentication details.

    Attributes:
        url: The webhook URL where notifications should be sent
        token: Optional unique token for this task/session
        authentication: Optional authentication configuration for secure notifications

    Example:
        ```python
        from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig
        from agentle.agents.a2a.models.authentication import Authentication

        # Create a simple notification configuration
        simple_config = PushNotificationConfig(
            url="https://example.com/webhooks/notifications",
            token="notification-token-123"
        )

        # Create a configuration with authentication
        auth = Authentication(
            schemes=["Bearer"],
            credentials="jwt_token_string"
        )

        secure_config = PushNotificationConfig(
            url="https://example.com/webhooks/secure-notifications",
            token="secure-notification-token-456",
            authentication=auth
        )
        ```

    Note:
        The URL should be a valid endpoint that can receive HTTP POST requests with
        JSON payloads. The token is used to identify the specific task or session
        that the notification is for.
    """

    url: str
    """The webhook URL where notifications should be sent"""

    token: str | None = Field(default=None)
    """Optional unique token for this task/session"""

    authentication: Authentication | None = Field(default=None)
    """Optional authentication configuration for secure notifications"""
