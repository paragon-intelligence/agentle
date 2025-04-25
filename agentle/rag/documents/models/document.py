from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from agentle.rag.documents.models.chunk import Chunk
from rsb.decorators.entities import entity


@dataclass
@entity
class Document:
    id: UUID
    chunks: Sequence[Chunk]
    metadata: Mapping[str, object]

    def describe(self) -> str:
        return f"<document>\n<chunks>\n{self.chunks}\n</chunks>\n<metadata>{self.metadata}</metadata>\n</document>"
