from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.messages.message import Message


class CallTranscript(BaseModel):
    """Complete call transcript with timing"""

    messages: list[Message] = Field(
        default_factory=list, description="Structured conversation messages"
    )
    raw_transcript: str = Field(default="", description="Raw transcript text")
    word_level_timestamps: list[dict[str, Any]] = Field(
        default_factory=list, description="Word-level timing data"
    )
    speaker_labels: dict[str, str] = Field(
        default_factory=dict, description="Speaker identification mapping"
    )
    confidence_scores: list[float] = Field(
        default_factory=list, description="Per-segment confidence scores"
    )
