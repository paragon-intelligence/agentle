from typing import Literal, Sequence

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.tools.tool import Tool


@valueobject
class DeveloperMessage(BaseModel):
    parts: Sequence[TextPart | FilePart | Tool]
    role: Literal["developer"] = Field(default="developer")
