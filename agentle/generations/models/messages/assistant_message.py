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
    parts: Sequence[TextPart | FilePart | ToolExecutionSuggestion]
    role: Literal["assistant"] = Field(default="assistant")
