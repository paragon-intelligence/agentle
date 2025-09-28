from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.real_time.definitions.audio_format import AudioFormat
from agentle.stt.real_time.definitions.language_code import LanguageCode


class STTConfig(BaseModel):
    """Configuration for STT providers."""

    language: LanguageCode = Field(default=LanguageCode.PT_BR)
    enable_word_timestamps: bool = Field(default=False)
    enable_automatic_punctuation: bool = Field(default=True)
    enable_speaker_diarization: bool = Field(default=False)
    model_name: str | None = Field(default=None, description="Provider-specific model")
    sample_rate: int = Field(default=16000)
    encoding: AudioFormat = Field(default=AudioFormat.PCM_S16LE)
    channels: int = Field(default=1)
    max_alternatives: int = Field(default=1, ge=1, le=10)
    profanity_filter: bool = Field(default=False)
    boost_phrases: list[str] | None = Field(
        default=None, description="Phrases to boost recognition"
    )
