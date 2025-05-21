"""
Module defining the UserMessage class representing messages from users.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, overload

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

    name: str | None = Field(
        default=None,
        description="The name of the user. If not provided, it will be set to 'User'.",
    )

    @overload
    def with_name_prepended(self, return_type: Literal["message"]) -> UserMessage: ...

    @overload
    def with_name_prepended(
        self, return_type: Literal["parts"]
    ) -> Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion]: ...

    def with_name_prepended(
        self, return_type: Literal["message", "parts"] = "message"
    ) -> (
        UserMessage
        | Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion]
    ):
        if return_type == "message":
            return UserMessage(
                role=self.role,
                parts=[TextPart(text=f"<name:{self.name}>")]
                + list(self.parts)
                + [TextPart(text="</name>")]
                if self.name
                else self.parts,
                name=self.name,
            )
        elif return_type == "parts":
            return (
                [TextPart(text=f"<name:{self.name}>")]
                + list(self.parts)
                + [TextPart(text="</name>")]
                if self.name
                else self.parts
            )
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
