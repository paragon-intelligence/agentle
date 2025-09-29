from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.transcriber_config import TranscriberConfig
from agentle.calls.voice_config import VoiceConfig


class RealtimeConversationProvider(BaseModel):
    """Unified provider that combines STT, TTS, and text generation for real-time conversations"""

    name: str = Field(description="Provider name")
    stt_provider: Any = Field(
        description="Speech-to-text provider instance"
    )  # RealtimeSpeechToTextProvider
    tts_provider: Any = Field(
        description="Text-to-speech provider instance"
    )  # RealtimeTextToSpeechProvider
    generation_provider: Any = Field(
        description="Text generation provider"
    )  # GenerationProvider

    # Configuration
    voice_config: VoiceConfig = Field(description="Voice synthesis configuration")
    transcriber_config: TranscriberConfig = Field(
        description="Speech recognition configuration"
    )

    # Performance settings
    target_latency_ms: int = Field(
        default=500, ge=100, le=2000, description="Target voice-to-voice latency"
    )
    enable_interruptions: bool = Field(
        default=True, description="Allow conversation interruptions"
    )
    conversation_buffer_size: int = Field(
        default=10, ge=1, le=50, description="Number of turns to keep in memory"
    )
