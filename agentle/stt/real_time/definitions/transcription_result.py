from rsb.models.base_model import BaseModel
from agentle.stt.real_time.definitions.language_code import LanguageCode
from agentle.stt.real_time.definitions.transcript_confidence import TranscriptConfidence
from rsb.models.field import Field
from agentle.stt.real_time.definitions.word_timing import WordTiming


class TranscriptionResult(BaseModel):
    """Result from speech-to-text transcription."""

    text: str
    language: LanguageCode
    confidence: TranscriptConfidence
    is_final: bool = Field(description="Whether this is a final or partial result")
    word_timings: list[WordTiming] | None = Field(default=None)
    audio_duration_ms: int | None = Field(default=None)
    processing_time_ms: float | None = Field(default=None)
