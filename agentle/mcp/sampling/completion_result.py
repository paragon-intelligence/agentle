"""
Completion Result Module

This module defines the CompletionResult class used to represent
the response from a language model after a completion request.
"""

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
    """
    Represents the result of a language model completion.

    This class encapsulates the response from a language model, including
    the model used, the reason why generation stopped, the role of the message,
    and the content of the message (which can be text or image).

    Attributes:
        model: The name of the model that generated the completion.
        stopReason: The reason why the generation stopped.
        role: The role associated with the generated content.
        content: The actual content of the completion, either text or image.
    """

    model: str = Field(
        description="Name of the model used for generation",
        examples=["gpt-4", "claude-3-opus"],
    )
    stopReason: Literal["endTurn", "stopSequence", "maxTokens"] | str | None = Field(
        default=None,
        description="Reason why the generation stopped, such as reaching end of turn, encountering a stop sequence, or hitting the maximum token limit",
    )
    role: Literal["user", "assistant"] = Field(
        description="Role associated with the message content, either 'user' or 'assistant'"
    )
    content: TextMessageContent | ImageMessageContent = Field(
        description="Content of the completion, which can be either text or image data",
    )
