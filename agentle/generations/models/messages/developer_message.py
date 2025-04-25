from typing import Literal, Sequence

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_declaration import ToolDeclaration


@valueobject
class DeveloperMessage(BaseModel):
    parts: Sequence[TextPart | FilePart | ToolDeclaration]
    role: Literal["developer"] = Field(default="developer")
