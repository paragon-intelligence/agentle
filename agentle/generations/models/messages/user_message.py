from collections.abc import Sequence
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_declaration import ToolDeclaration
from agentle.generations.models.message_parts.tool_execution import ToolExecution


class UserMessage(BaseModel):
    parts: Sequence[TextPart | FilePart | ToolDeclaration | ToolExecution]
    role: Literal["user"] = Field(default="user")
