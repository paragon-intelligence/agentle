from typing import Annotated

from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_declaration import ToolDeclaration
from agentle.generations.models.message_parts.tool_execution import ToolExecution

type Part = Annotated[
    TextPart | FilePart | ToolDeclaration | ToolExecution, Field(discriminator="type")
]
