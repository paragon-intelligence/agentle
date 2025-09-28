
from rsb.models.base_model import BaseModel

from agentle.tts.real_time.definitions.audio_data import AudioData


class SpeechResult(BaseModel):
    """Result from text-to-speech synthesis."""

    audio: AudioData
    text: str
    voice_id: str
    processing_time_ms: float
    audio_duration_ms: int
    character_count: int
