from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai.types import Part as GooglePart


class PartToGooglePartAdapter(
    Adapter[TextPart | FilePart | ToolExecutionSuggestion | Tool, "GooglePart"]
):
    def adapt(
        self, _f: TextPart | FilePart | ToolExecutionSuggestion | Tool
    ) -> GooglePart:
        from google.genai.types import Blob, FunctionCall
        from google.genai.types import Part as GooglePart

        match _f:
            case TextPart():
                return GooglePart(text=_f.text)
            case FilePart():
                return GooglePart(
                    inline_data=Blob(data=_f.data, mime_type=_f.mime_type)
                )
            case ToolExecutionSuggestion():
                return GooglePart(
                    function_call=FunctionCall(
                        id=_f.id,
                        name=_f.tool_name,
                        args=_f.args,
                    )
                )
            case Tool():
                return GooglePart(text=f"<tool>{_f.name}</tool>")
