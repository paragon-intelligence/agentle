import base64

from rsb.adapters.adapter import Adapter

from agentle.agents.a2a.message_parts.file_part import FilePart
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.models.file import File
from agentle.generations.models.message_parts.file import (
    FilePart as GenerationFilePart,
)
from agentle.generations.models.message_parts.text import TextPart as GenerationTextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion as GenerationToolExecutionSuggestion,
)
from agentle.generations.tools.tool import (
    Tool as GenerationTool,
)


class GenerationPartToAgentPartAdapter(
    Adapter[
        GenerationFilePart
        | GenerationTextPart
        | GenerationTool
        | GenerationToolExecutionSuggestion,
        FilePart | TextPart,
    ]
):
    def adapt(
        self,
        _f: GenerationFilePart
        | GenerationTextPart
        | GenerationTool
        | GenerationToolExecutionSuggestion,
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
            case GenerationTool():
                raise NotImplementedError("Tool declarations are not supported")
            case GenerationToolExecutionSuggestion():
                raise NotImplementedError("Tool executions are not supported")
