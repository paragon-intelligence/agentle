from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Never

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.part import Part
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)

if TYPE_CHECKING:
    from google.genai.types import Part as GooglePart


class GooglePartToPartAdapter(Adapter["GooglePart", Part]):
    def adapt(self, _f: GooglePart) -> Part:
        if _f.text:
            return TextPart(text=_f.text)

        if _f.inline_data:
            data = _f.inline_data.data or self._raise_invalid_inline_data(field="data")
            mime_type = _f.inline_data.mime_type or self._raise_invalid_inline_data(
                field="mime_type"
            )
            return FilePart(data=data, mime_type=mime_type)

        if _f.function_call:
            return ToolExecutionSuggestion(
                id=_f.function_call.id or str(uuid.uuid4()),
                tool_name=_f.function_call.name
                or self._raise_invalid_function_call(field="name"),
                args=_f.function_call.args
                or self._raise_invalid_function_call(field="args"),
            )

        raise ValueError(
            f"The provided part: {_f} is not supported by the framework yet."
        )

    def _raise_invalid_inline_data(self, field: str) -> Never:
        raise ValueError(f"Provided field '{field}' is None.")

    def _raise_invalid_function_call(self, field: str) -> Never:
        raise ValueError(f"Provided field '{field}' is None.")
