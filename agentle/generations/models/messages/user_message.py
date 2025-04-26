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
    parts: Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion]
    role: Literal["user"] = Field(default="user")
