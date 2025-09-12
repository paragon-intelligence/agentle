from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WordTiming(BaseModel):
    """Word-level timing information."""

    word: str
    start_time_ms: int
    end_time_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
