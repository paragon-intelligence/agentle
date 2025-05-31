from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload from WhatsApp."""

    # Evolution API
    event_type: str | None = Field(default=None)
    instance_id: str | None = Field(default=None)
    data: Mapping[str, Any] | None = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Meta WhatsApp Business API
    entry: list[dict[str, Any]] | None = Field(default=None)
    changes: list[dict[str, Any]] | None = Field(default=None)
    field: str | None = Field(default=None)
    value: Mapping[str, Any] | None = Field(default=None)
    phone_number_id: str | None = Field(default=None)
    metadata: Mapping[str, Any] | None = Field(default=None)
    status: str | None = Field(default=None)
    status_code: int | None = Field(default=None)
