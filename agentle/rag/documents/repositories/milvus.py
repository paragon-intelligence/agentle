from typing import override

from agentle.rag.documents.models.document import Document
from rsb.contracts.repositories.readable import AsyncReader


class MilvusRagDocumentRepository(AsyncReader[Document, str]):
    @override
    async def read(
        self, uid: str, filters: dict[str, object] | None = None
    ) -> Document: ...
