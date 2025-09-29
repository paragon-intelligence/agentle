from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.tts_provider import TTSProvider


class VoiceConfig(BaseModel):
    """Text-to-Speech configuration"""

    provider: TTSProvider = Field(default=TTSProvider.ELEVENLABS)
    voice_id: str = Field(description="Voice identifier from the provider")
    speed: float = Field(
        default=1.0, ge=0.1, le=3.0, description="Speech speed multiplier"
    )
    stability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice stability (provider-specific)"
    )
    similarity: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Voice similarity (provider-specific)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Voice creativity/randomness"
    )
    use_speaker_boost: bool = Field(default=True, description="Boost speaker clarity")
    optimize_streaming_latency: bool = Field(
        default=True, description="Optimize for real-time streaming"
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific settings"
    )
