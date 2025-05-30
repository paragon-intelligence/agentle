from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, override
from uuid import UUID

from agentle.rag.documents.models.blob import Blob
from agentle.rag.documents.models.chunk import Chunk
from agentle.rag.documents.models.document import Document
from agentle.rag.documents.repositories.downloadable import AsyncDownloadable
from rsb.contracts.repositories.deletable import AsyncDeletable
from rsb.contracts.repositories.readable import (
    AsyncBulkReader,
    AsyncReader,
)
from rsb.contracts.repositories.writable import AsyncWritable

if TYPE_CHECKING:
    from r2r import R2RAsyncClient


class R2RGraphDocumentRepository(
    AsyncReader[Document],
    AsyncBulkReader[Document],
    AsyncWritable[Path],
    AsyncDeletable[str],
    AsyncDownloadable[Blob],
):
    r2r_client: R2RAsyncClient

    def __init__(self, r2r_client: R2RAsyncClient):
        self.r2r_client = r2r_client

    @override
    async def read_async(
        self, uid: str, filters: dict[str, object] | None = None
    ) -> Document:
        from sdk.models import GraphSearchSettings, SearchSettings

        document_response = await self.r2r_client.documents.list(ids=[uid])
        if len(document_response.results) == 0:
            raise ValueError("Document not found in the documents database.")

        if filters is None:
            raise ValueError(
                "Filters is required to access the `read` method from R2RRepository"
            )

        query = filters.get("query")
        if query is None:
            raise ValueError("Query must be informed by the user.")

        r2r_search_result = await self.r2r_client.retrieval.search(
            query=query,
            search_settings=SearchSettings(
                graph_settings=GraphSearchSettings(enabled=True),
                num_sub_queries=filters.get("num_sub_queries") or 5,
                search_strategy="query_fusion",
                limit=filters.get("limit") or 10,
                filters={"document_id": {"$eq": uid}},
            ),
        )

        chunk_search_results = r2r_search_result.results.chunk_search_results
        if chunk_search_results is None:
            return Document(id=UUID(uid), chunks=[], metadata={})

        chunks: list[Chunk] = [
            Chunk(
                id=c.id,
                document_id=c.document_id,
                score=c.score,
                text=c.text,
                metadata=c.metadata,
            )
            for c in chunk_search_results
        ]

        return Document(id=UUID(uid), chunks=chunks, metadata={})

    @override
    async def read_all_async(
        self, filters: dict[str, object] | None = None
    ) -> Sequence[Document]:
        from sdk.models import GraphSearchSettings, SearchSettings

        if filters is None:
            raise ValueError(
                "Filters is required to access the `read_all` method from R2RRepository"
            )

        query = filters.get("query")
        if query is None:
            raise ValueError("Query must be informed by the user.")

        r2r_search_result = await self.r2r_client.retrieval.search(
            query=query.__str__(),
            search_settings=SearchSettings(
                graph_settings=GraphSearchSettings(
                    enabled=bool(filters.get("graph_search_enabled")) or True,
                ),
                num_sub_queries=filters.get("num_sub_queries") or 5,
                search_strategy=filters.get("search_strategy") or "query_fusion",
                limit=filters.get("limit") or 20,
                filters=filters.get("filters") or {},
            ),
        )

        chunk_search_results = r2r_search_result.results.chunk_search_results

        if chunk_search_results is None:
            return []

        chunks: list[Chunk] = []
        unique_document_ids: set[UUID] = set([])
        document_ids_to_related_chunks: dict[UUID, set[Chunk]] = {}

        for chunk in chunk_search_results:
            _chunk = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text,
                metadata=chunk.metadata,
            )

            chunks.append(_chunk)

            unique_document_ids.add(chunk.id)
            document_ids_to_related_chunks[chunk.document_id].add(_chunk)

        documents: list[Document] = [
            Document(
                id=document_id,
                chunks=list(document_ids_to_related_chunks[document_id]),
                metadata={},
            )
            for document_id in unique_document_ids
        ]

        return documents

    @override
    async def delete_async(self, uid: str) -> None:
        await self.r2r_client.documents.delete(uid)

    @override
    async def download_async(self, uid: str) -> Blob:
        download = await self.r2r_client.documents.download(uid)
        return Blob(data=download.read(), extension="")

    async def write(self, e: Path) -> None:
        await self.r2r_client.documents.create(file_path=str(e))
