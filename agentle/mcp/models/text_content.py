"""
Text Content Module for MCP.

This module defines the TextContent class which represents textual content
in the Model Control Protocol (MCP) system. It provides a structure for
including plain text within messages or responses.
"""

from typing import Literal

from agentle.mcp.models.annotations import Annotations
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class TextContent(BaseModel):
    """
    Text content for a message in the MCP system.

    TextContent provides a structure for representing textual data within
    messages or responses. This is the most common content type used for
    typical conversational exchanges. Optional annotations can be included
    to provide additional context or metadata about the text.

    Attributes:
        type (Literal["text"]): The content type identifier, always "text"
        text (str): The textual content of the message
        annotations (Annotations | None): Optional annotations for this text
    """

    type: Literal["text"] = Field(
        default="text", description="Content type identifier, always 'text'"
    )
    text: str = Field(..., description="The textual content of the message")
    annotations: Annotations | None = Field(
        default=None, description="Optional annotations for this text"
    )
    model_config = ConfigDict(extra="allow")
