"""
Module defining the UserMessage class representing messages from users.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool


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

    @classmethod
    def create_named(
        cls,
        parts: Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion],
        name: str | None = None,
    ) -> UserMessage:
        """
        Creates a user message with a name tag wrapped around the parts.

        Args:
            parts: The sequence of message parts to include in the message.
            name: Optional name to tag this message with. If provided, the parts will
                  be wrapped with <name:{name}> and </name:{name}> tags.

        Returns:
            A UserMessage instance with the parts wrapped in name tags if a name is provided.
        """
        return cls(
            role="user",
            parts=[TextPart(text=f"<name:{name}>")]
            + list(parts)
            + [TextPart(text=f"</name:{name}>")],
        )
