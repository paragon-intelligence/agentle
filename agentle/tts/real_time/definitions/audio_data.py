from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.stt.real_time.definitions.audio_format import AudioFormat


class AudioData(BaseModel):
    """Audio data with metadata."""

    data: bytes
    format: AudioFormat
    sample_rate: int = Field(description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    bit_depth: int = Field(default=16, description="Bit depth")
    duration_ms: int | None = Field(
        default=None, description="Duration in milliseconds"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
