from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.real_time.definitions.audio_data import AudioData


class STTStreamChunk(BaseModel):
    """Chunk of streaming STT data."""

    audio_chunk: AudioData
    sequence_number: int
    is_final_chunk: bool = Field(default=False)
