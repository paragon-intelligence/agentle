from typing import Literal

from agentle.mcp.sampling.messages.image_message_content import (
    ImageMessageContent,
)
from agentle.mcp.sampling.messages.text_message_content import (
    TextMessageContent,
)
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class AssistantMessage(BaseModel):
    content: TextMessageContent | ImageMessageContent = Field(discriminator="type")
    role: Literal["assistant"] = Field(default="assistant")
