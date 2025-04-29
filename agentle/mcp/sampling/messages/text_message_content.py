"""
Text Message Content module for MCP.

This module defines the TextMessageContent class, which represents text-based
content in messages within the Model Control Protocol (MCP) system. It provides
a structure for including text data within messages.
"""

from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TextMessageContent(BaseModel):
    """
    Text content for messages in the MCP system.

    TextMessageContent represents the content of a message that contains text.
    It stores the text string and includes a type field that is always set to "text"
    to identify this content type.

    Attributes:
        text (str | None): The text content of the message
        type (Literal["text"]): The content type, always set to "text"
    """

    text: str | None = Field(description="Text content of the message")
    type: Literal["text"] = Field(
        default="text",
        description="Content type identifier, always 'text' for this class",
    )
