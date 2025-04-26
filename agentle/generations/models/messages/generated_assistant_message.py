from collections.abc import Sequence
from typing import Literal

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


@valueobject
class GeneratedAssistantMessage[T](BaseModel):
    parts: Sequence[TextPart | ToolExecutionSuggestion]
    parsed: T
    role: Literal["assistant"] = Field(default="assistant")

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
        return [
            part for part in self.parts if isinstance(part, ToolExecutionSuggestion)
        ]

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts)
