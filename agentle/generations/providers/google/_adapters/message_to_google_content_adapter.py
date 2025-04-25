from __future__ import annotations

from typing import TYPE_CHECKING, override

from agentle.generations.models.messages.message import Message
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.google._adapters.part_to_google_part_adapter import (
    PartToGooglePartAdapter,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from google.genai.types import Content


class MessageToGoogleContentAdapter(Adapter[Message, "Content"]):
    part_adapter: PartToGooglePartAdapter

    def __init__(self, part_adapter: PartToGooglePartAdapter | None = None) -> None:
        super().__init__()
        self.part_adapter = part_adapter or PartToGooglePartAdapter()

    @override
    def adapt(self, _f: Message) -> Content:
        from google.genai.types import Content

        part_adapter = self.part_adapter or PartToGooglePartAdapter()

        match _f.role:
            case "assistant":
                role = "model"
            case "developer":
                role = "model"
            case "user":
                role = "user"

        return Content(
            parts=[
                part_adapter.adapt(part)
                for part in _f.parts
                if not isinstance(  # Impossible that an AI declares a tool
                    part, Tool
                )
            ],
            role=role,
        )
