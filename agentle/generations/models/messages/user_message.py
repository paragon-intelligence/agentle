"""
Module defining the UserMessage class representing messages from users.
"""

from collections.abc import Sequence
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.tools.tool import Tool
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class UserMessage(BaseModel):
    """
    Represents a message from a user in the system.

    This class can contain a sequence of different message parts including
    text, files, tools, and tool execution suggestions.
    """

    role: Literal["user"] = Field(
        default="user",
        description="Discriminator field to identify this as a user message. Always set to 'user'.",
    )

    parts: Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion] = Field(
        description="The sequence of message parts that make up this user message.",
    )
