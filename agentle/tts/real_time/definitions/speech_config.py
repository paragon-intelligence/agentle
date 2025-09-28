from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.real_time.definitions.audio_format import AudioFormat
from agentle.stt.real_time.definitions.language_code import LanguageCode


class SpeechConfig(BaseModel):
    """Configuration for speech synthesis."""

    voice_id: str
    language: LanguageCode = Field(default=LanguageCode.PT_BR)
    speed: float = Field(
        default=1.0, ge=0.1, le=3.0, description="Speech rate multiplier"
    )
    pitch: float = Field(default=1.0, ge=0.1, le=2.0, description="Pitch multiplier")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Volume multiplier")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV)
    sample_rate: int = Field(default=22050)
    enable_ssml: bool = Field(
        default=False, description="Whether text contains SSML markup"
    )
    emotion: str | None = Field(default=None, description="Emotional tone")
    stability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Voice stability"
    )
    similarity_boost: float | None = Field(default=None, ge=0.0, le=1.0)
