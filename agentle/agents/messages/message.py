from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.message_parts.data_part import DataPart
from agentle.agents.message_parts.file_part import FilePart
from agentle.agents.message_parts.text_part import TextPart


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: Sequence[TextPart | FilePart | DataPart]
    metadata: dict[str, Any] | None = Field(default=None)
