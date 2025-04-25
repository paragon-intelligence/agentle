from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from rsb.decorators.value_objects import valueobject


@dataclass(frozen=True)
@valueobject
class Chunk:
    id: UUID
    document_id: UUID
    score: float | None
    text: str
    metadata: Mapping[str, object]

    def describe(self) -> str:
        return f"<chunk>\n<text>\n{self.text}\n</text>\n<metadata>{self.metadata}\n</metadata>\n</chunk>"
