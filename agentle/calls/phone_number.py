from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.transport_provider import TransportProvider


class PhoneNumber(BaseModel):
    """Represents a phone number with provider configuration"""

    number: str = Field(description="Phone number in E.164 format (e.g., +1234567890)")
    provider: TransportProvider = Field(default=TransportProvider.CUSTOM_SIP)
    provider_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )
    country_code: str | None = Field(default=None, description="ISO country code")
    is_verified: bool = Field(
        default=False, description="Whether number is verified for outbound calls"
    )
    call_limit_per_day: int | None = Field(
        default=None, description="Daily call limit for this number"
    )

    def validate_e164(self) -> bool:
        """Validate E.164 format"""
        return self.number.startswith("+") and self.number[1:].isdigit()
