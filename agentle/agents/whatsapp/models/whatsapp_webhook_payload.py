from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.data import Data


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload from WhatsApp."""

    # Evolution API
    event: str | None = Field(default=None)
    instance: str | None = Field(default=None)
    data: Data | None = Field(default=None)
    destination: str | None = Field(default=None)
    date_time: datetime | None = Field(default=None)
    sender: str | None = Field(default=None)
    server_url: str | None = Field(default=None)
    apikey: str | None = Field(default=None)

    # Meta WhatsApp Business API
    entry: list[dict[str, Any]] | None = Field(default=None)
    changes: list[dict[str, Any]] | None = Field(default=None)
    field: str | None = Field(default=None)
    value: Mapping[str, Any] | None = Field(default=None)
    phone_number_id: str | None = Field(default=None)
    metadata: Mapping[str, Any] | None = Field(default=None)
    status: str | None = Field(default=None)
    status_code: int | None = Field(default=None)
