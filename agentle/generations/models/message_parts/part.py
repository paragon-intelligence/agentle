"""
Module defining the Part type that represents different types of message parts.

This module provides a unified type for different message parts like text, files,
tools, and tool execution suggestions using Annotated typing.
"""

from typing import Annotated

from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool

# Part is a union type that can be any of the message part types
# The Field with discriminator indicates that the "type" field is used
# to determine which concrete class to use when deserializing
type Part = Annotated[
    TextPart | FilePart | Tool | ToolExecutionSuggestion, Field(discriminator="type")
]
