import uuid
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.messages.message import Message
from agentle.agents.notifications.push_notification_config import PushNotificationConfig


class TaskSendParams(BaseModel):
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
    historyLength: int | None = Field(default=None)
    pushNotification: PushNotificationConfig | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)
