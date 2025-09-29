from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.stt_provider import STTProvider
from agentle.stt.real_time.definitions.language_code import LanguageCode


class TranscriberConfig(BaseModel):
    """Speech-to-Text configuration"""

    provider: STTProvider = Field(default=STTProvider.DEEPGRAM)
    model: str = Field(default="nova-2", description="STT model to use")
    language: LanguageCode = Field(default=LanguageCode.EN_US)
    smart_format: bool = Field(default=True, description="Apply intelligent formatting")
    profanity_filter: bool = Field(
        default=False, description="Filter profanity from transcripts"
    )
    redact_pii: bool = Field(
        default=False, description="Redact personally identifiable information"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords to boost recognition"
    )
    endpointing_ms: int = Field(
        default=200,
        ge=100,
        le=2000,
        description="Silence duration to detect speech end",
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific settings"
    )
