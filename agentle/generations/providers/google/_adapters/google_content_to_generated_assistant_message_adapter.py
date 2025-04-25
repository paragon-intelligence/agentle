from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.providers.google._adapters.google_part_to_part_adapter import (
    GooglePartToPartAdapter,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from google.genai.types import Content


class GoogleContentToGeneratedAssistantMessageAdapter[T](
    Adapter["Content", GeneratedAssistantMessage[T]]
):
    part_adapter: GooglePartToPartAdapter
    generate_content_response_parsed: T | None

    def __init__(
        self,
        part_adapter: GooglePartToPartAdapter | None = None,
        generate_content_response_parsed: T | None = None,
    ) -> None:
        super().__init__()
        self.part_adapter = part_adapter or GooglePartToPartAdapter()
        self.generate_content_response_parsed = generate_content_response_parsed

    @override
    def adapt(self, _f: Content) -> GeneratedAssistantMessage[T]:
        parts = _f.parts

        if parts is None:
            raise ValueError("No parts found in Google Content.")

        adapted_parts = [self.part_adapter.adapt(part) for part in parts]

        role = _f.role
        if role is None:
            raise ValueError("No role found in Google Content.")

        match role:
            case "model":
                message = GeneratedAssistantMessage[T](
                    parts=adapted_parts,
                    parsed=self.generate_content_response_parsed
                    if self.generate_content_response_parsed
                    else cast(T, None),
                )

                return message
            case _:
                raise ValueError(
                    f"This adapter does only supports assistant messages. Provided: {_f}"
                )
