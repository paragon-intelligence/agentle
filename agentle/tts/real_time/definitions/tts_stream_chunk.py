from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.tts.real_time.definitions.audio_data import AudioData


class TTSStreamChunk(BaseModel):
    """Chunk of streaming TTS data."""

    audio_chunk: AudioData
    sequence_number: int
    is_final_chunk: bool = Field(default=False)
    text_processed: str | None = Field(
        default=None, description="Text that generated this chunk"
    )
