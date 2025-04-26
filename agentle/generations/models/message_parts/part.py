from typing import Annotated

from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool

type Part = Annotated[
    TextPart | FilePart | Tool | ToolExecutionSuggestion, Field(discriminator="type")
]
