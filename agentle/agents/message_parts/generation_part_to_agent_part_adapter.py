import base64

from rsb.adapters.adapter import Adapter

from agentle.agents.message_parts.file_part import FilePart
from agentle.agents.message_parts.text_part import TextPart
from agentle.agents.models.file import File
from agentle.generations.models.message_parts.file import (
    FilePart as GenerationFilePart,
)
from agentle.generations.models.message_parts.text import TextPart as GenerationTextPart
from agentle.generations.models.message_parts.tool_declaration import (
    ToolDeclaration as GenerationToolDeclaration,
)
from agentle.generations.models.message_parts.tool_execution import (
    ToolExecution as GenerationToolExecution,
)


class GenerationPartToAgentPartAdapter(
    Adapter[
        GenerationFilePart
        | GenerationTextPart
        | GenerationToolDeclaration
        | GenerationToolExecution,
        FilePart | TextPart,
    ]
):
    def adapt(
        self,
        _f: GenerationFilePart
        | GenerationTextPart
        | GenerationToolDeclaration
        | GenerationToolExecution,
    ) -> FilePart | TextPart:
        match _f:
            case GenerationFilePart():
                return FilePart(
                    type=_f.type,
                    file=File(
                        bytes=base64.b64encode(_f.data).decode("utf-8"),
                    ),
                )
            case GenerationTextPart():
                return TextPart(text=_f.text)
            case GenerationToolDeclaration():
                raise NotImplementedError("Tool declarations are not supported")
            case GenerationToolExecution():
                raise NotImplementedError("Tool executions are not supported")
