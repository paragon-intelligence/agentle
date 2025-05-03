from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TranscriptionConfig(BaseModel):
    """Configuration for transcription."""

    timeout: float = Field(
        default=50.0,
        description="The timeout for the transcription in seconds.",
    )

    context_prompt: str = Field(
        default="",
        description="A prompt to provide context for the transcription.",
    )
