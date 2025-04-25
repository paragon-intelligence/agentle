from typing import Literal

from agentle.mcp.sampling.messages.image_message_content import (
    ImageMessageContent,
)
from agentle.mcp.sampling.messages.text_message_content import (
    TextMessageContent,
)
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class CompletionResult(BaseModel):
    model: str = Field(description="Name of the model used")
    stopReason: Literal["endTurn", "stopSequence", "maxTokens"] | str | None = Field(
        default=None
    )
    role: Literal["user", "assistant"]
    content: TextMessageContent | ImageMessageContent = Field(discriminator="type")
