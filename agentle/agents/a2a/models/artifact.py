from collections.abc import Sequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.generations.models.message_parts.text import TextPart


class Artifact(BaseModel):
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    parts: Sequence[TextPart]
    metadata: dict[str, Any] | None = Field(default=None)
    index: int
    append: bool | None = Field(default=None)
    last_chunk: bool | None = Field(default=None)

    model_config = ConfigDict(frozen=True)
