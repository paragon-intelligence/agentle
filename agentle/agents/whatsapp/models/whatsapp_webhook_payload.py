from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload from WhatsApp."""

    event_type: str
    instance_id: str
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
