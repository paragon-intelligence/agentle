from __future__ import annotations

from agentle.rag.documents.models.chunk import Chunk
from rsb.collections.readonly_collection import ReadonlyCollection


class ChunkSequence(ReadonlyCollection[Chunk]):
    def describe(self) -> str:
        return "".join(chunk.describe() for chunk in self.elements)
