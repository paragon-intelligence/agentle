from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TranscriptionConfig(BaseModel):
    """Configuration for transcription."""

    timeout: float = Field(
        default=50.0,
        description="The timeout for the transcription in seconds.",
    )

    language: str = Field(
        default="en",
        description="The language of the transcription.",
    )

    temperature: float = Field(
        default=0.0,
        description="The temperature for the transcription.",
    )

    context_prompt: str = Field(
        default="",
        description="A prompt to provide context for the transcription.",
    )
