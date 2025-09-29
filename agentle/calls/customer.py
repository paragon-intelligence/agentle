import uuid
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.real_time.definitions.language_code import LanguageCode


class Customer(BaseModel):
    """Represents a call recipient"""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique customer identifier",
    )
    phone_number: str = Field(description="Customer phone number in E.164 format")
    name: str | None = Field(default=None, description="Customer name")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional customer data"
    )
    timezone: str | None = Field(
        default=None, description="Customer timezone (e.g., America/New_York)"
    )
    preferred_language: LanguageCode | None = Field(
        default=None, description="Customer's preferred language"
    )
