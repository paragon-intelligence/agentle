from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from agentle.rag.documents.models.document import Document
from rsb.contracts.repositories.readable import AsyncBulkReader
from rsb.coroutines.run_sync import run_sync
from rsb.decorators.services import abstractservice


@abstractservice
class SimilarDocumentsRetrievalServiceProtocol:
    repository: AsyncBulkReader[Document]

    def __init__(self, repository: AsyncBulkReader[Document]) -> None:
        self.repository = repository

    def retrieve_similar(self, query: str) -> Sequence[Document]:
        return run_sync(self.retrieve_similar_async, timeout=None, query=query)

    @abstractmethod
    async def retrieve_similar_async(self, query: str) -> Sequence[Document]: ...
