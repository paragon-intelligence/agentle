from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TranscriptConfidence(BaseModel):
    """Confidence metrics for transcription."""

    overall: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    word_level: list[float] | None = Field(
        default=None, description="Per-word confidence scores"
    )
