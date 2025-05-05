"""
Module defining the AssistantMessage class representing messages from assistants.
"""

from typing import Literal, Sequence

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class AssistantMessage(BaseModel):
    """
    Represents a message from an assistant in the system.

    This class can contain a sequence of different message parts including
    text, files, and tool execution suggestions.
    """

    role: Literal["assistant"] = Field(
        default="assistant",
        description="Discriminator field to identify this as an assistant message. Always set to 'assistant'.",
    )

    parts: Sequence[TextPart | FilePart | ToolExecutionSuggestion] = Field(
        description="The sequence of message parts that make up this assistant message.",
    )
