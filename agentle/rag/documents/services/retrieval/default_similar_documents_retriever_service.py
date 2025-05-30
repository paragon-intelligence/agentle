from __future__ import annotations

from collections.abc import Sequence
from typing import override

from agentle.rag.documents.models.document import Document
from agentle.rag.documents.services.retrieval.similar_documents_retrieval_service import (
    SimilarDocumentsRetrievalServiceProtocol,
)


class DefaultSimilarDocumentsRetrieverService(SimilarDocumentsRetrievalServiceProtocol):
    @override
    async def retrieve_similar_async(self, query: str) -> Sequence[Document]:
        return await self.repository.read_all_async({"query": query})
