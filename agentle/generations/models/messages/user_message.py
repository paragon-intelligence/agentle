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
        Creates a user message with a name identifier.

        Args:
            parts: The sequence of message parts to include in the message.
            name: Optional name to identify the user sending this message.

        Returns:
            A UserMessage instance with the name prepended if provided.
        """
        if name is None:
            return cls(role="user", parts=parts)
        
        return cls(
            role="user",
            parts=[TextPart(text=f"[{name}]: ")] + list(parts),
        )
