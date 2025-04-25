from collections.abc import Sequence
from typing import Literal

from agentle.generations.models.message_parts.part import Part
from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class GeneratedAssistantMessage[T](BaseModel):
    parts: Sequence[Part]
    parsed: T
    role: Literal["assistant"] = Field(default="assistant")

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts)
