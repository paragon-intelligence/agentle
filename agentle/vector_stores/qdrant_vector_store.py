from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, MutableSequence
from typing import TYPE_CHECKING, Any, Sequence, override

from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.parsing.chunk import Chunk
from agentle.vector_stores.collection import Collection
from agentle.vector_stores.create_collection_config import CreateCollectionConfig
from agentle.vector_stores.filters.filter import Filter
from agentle.vector_stores.vector_store import VectorStore

if TYPE_CHECKING:
    from qdrant_client.async_qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    _client: AsyncQdrantClient

    def __init__(
        self,
        *,
        default_collection_name: str = "agentle",
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProviderType | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: dict[str, Any] | None = None,
        auth_token_provider: Callable[[], str]
        | Callable[[], Awaitable[str]]
        | None = None,
        cloud_inference: bool = False,
        local_inference_batch_size: int | None = None,
        check_compatibility: bool = True,
        detailed_agent_description: str | None = None,
    ) -> None:
        from qdrant_client.async_qdrant_client import AsyncQdrantClient

        super().__init__(
            default_collection_name=default_collection_name,
            embedding_provider=embedding_provider,
            generation_provider=generation_provider,
            detailed_agent_description=detailed_agent_description,
        )

        self._client = AsyncQdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
            grpc_options=grpc_options,
            auth_token_provider=auth_token_provider,
            cloud_inference=cloud_inference,
            local_inference_batch_size=local_inference_batch_size,
            check_compatibility=check_compatibility,
        )

    @override
    async def _find_related_content_async(
        self,
        query: Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]:
        from qdrant_client.http.models.models import Filter as QdrantFilter

        chunks: MutableSequence[Chunk] = []

        if query is None:
            if filter is None:
                raise ValueError("ERROR: query or field must be provided.")

            should = filter.should
            if should is not None:
                if isinstance(should, Sequence):
                    should = [s.to_qdrant_condition() for s in should]
                else:
                    should = should.to_qdrant_condition()

            min_should = filter.min_should
            if min_should is not None:
                min_should = min_should.to_qdrant_min_should()

            must = filter.must
            if must is not None:
                if isinstance(must, Sequence):
                    must = [s.to_qdrant_condition() for s in must]
                else:
                    must = must.to_qdrant_condition()

            must_not = filter.must_not
            if must_not is not None:
                if isinstance(must_not, Sequence):
                    must_not = [s.to_qdrant_condition() for s in must_not]
                else:
                    must_not = must_not.to_qdrant_condition()

            filter_response = await self._client.query_points(
                collection_name=collection_name or self.default_collection_name,
                query_filter=QdrantFilter(
                    should=should,
                    min_should=min_should,
                    must=must,
                    must_not=must_not,
                ),
            )

            for scored_point in filter_response.points:
                payload = scored_point.payload
                if payload is None:
                    raise RuntimeError(
                        "Could not load Payload. Payload is needed "
                        + "to decode the vector into text."
                    )
                text = payload.get("text")
                if text is None:
                    raise RuntimeError(
                        "Error: could not load payload text. "
                        + "Text is needed to get original "
                        + "payload contents"
                    )

                metadata: Mapping[str, Any] | None = payload.get("metadata")
                if metadata is not None and not isinstance(metadata, dict):
                    raise ValueError(
                        "Metadata is not an instance of Mapping and "
                        + "it's not None. It must be a Mapping."
                    )

                chunks.append(
                    Chunk(id=str(scored_point.id), text=text, metadata=metadata or {})
                )

            return chunks

        query_response = await self._client.query_points(
            collection_name=collection_name or self.default_collection_name,
            query=list(query),
            limit=k,
        )

        for scored_point in query_response.points:
            payload = scored_point.payload
            if payload is None:
                raise RuntimeError(
                    "Could not load Payload. Payload is needed "
                    + "to decode the vector into text."
                )
            text = payload.get("text")
            if text is None:
                raise RuntimeError(
                    "Error: could not load payload text. "
                    + "Text is needed to get original "
                    + "payload contents"
                )

            metadata: Mapping[str, Any] | None = payload.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    "Metadata is not an instance of Mapping and "
                    + "it's not None. It must be a Mapping."
                )

            chunks.append(
                Chunk(id=str(scored_point.id), text=text, metadata=metadata or {})
            )

        return chunks

    @override
    async def create_collection_async(
        self, collection_name: str, *, config: CreateCollectionConfig
    ) -> None:
        from qdrant_client.http.models.models import Distance, VectorParams

        collection_exists = await self._client.collection_exists(
            collection_name=collection_name
        )

        if collection_exists:
            logger.debug("collection exists. skipping creation.")
            return None

        result = await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config["size"],
                distance={
                    "COSINE": Distance.COSINE,
                    "EUCLID": Distance.EUCLID,
                    "DOT": Distance.DOT,
                    "MANHATTAN": Distance.MANHATTAN,
                }[config["distance"]],
            ),
        )

        if not result:
            raise RuntimeError("Unable to create collection")

    @override
    async def delete_collection_async(self, collection_name: str) -> None:
        result = await self._client.delete_collection(collection_name=collection_name)
        if not result:
            raise ValueError(
                "Unable to delete collection name from the database. Verify "
                + "your Qdrant instance or collection name provided."
            )

        return

    @override
    async def list_collections_async(self) -> Sequence[Collection]:
        from qdrant_client.conversions import common_types as types

        collections_response = await self._client.get_collections()
        collections: MutableSequence[Collection] = []

        for qdrant_collection in collections_response.collections:
            collection_info: types.CollectionInfo = await self._client.get_collection(
                collection_name=qdrant_collection.name
            )

            collections.append(
                Collection(
                    name=qdrant_collection.name,
                    indexed_vectors_count=collection_info.indexed_vectors_count,
                )
            )

        return collections

    @override
    async def _upsert_async(
        self,
        points: Embedding,
        *,
        collection_name: str | None = None,
    ) -> None:
        from qdrant_client.http.models.models import (
            PointStruct,
        )

        extra_metadata = {}
        if points.metadata.get("source_document_id"):
            extra_metadata["source_document_id"] = points.metadata["source_document_id"]

        await self._client.upsert(
            collection_name=collection_name or self.default_collection_name,
            points=[
                PointStruct(
                    id=points.id,
                    vector=list(points.value),
                    payload={
                        "text": points.original_text,
                        "metadata": points.metadata,
                        **extra_metadata,
                    },
                )
            ],
        )

    @override
    async def _delete_vectors_async(
        self,
        collection_name: str,
        ids: Sequence[str],
    ) -> None:
        await self._client.delete(
            collection_name=collection_name, points_selector=list(ids)
        )


if __name__ == "__main__":
    from pprint import pprint

    from agentle.embeddings.providers.google.google_embedding_provider import (
        GoogleEmbeddingProvider,
    )
    from agentle.parsing.parsers.pdf import PDFFileParser
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG)

    qdrant = QdrantVectorStore(
        embedding_provider=GoogleEmbeddingProvider(
            vertexai=True, project="unicortex", location="global"
        )
    )

    # qdrant.delete_collection("test_collection")

    qdrant.create_collection(
        "test_collection", config={"size": 3072, "distance": "COSINE"}
    )

    pprint(qdrant.list_collections())

    pdf_parser = PDFFileParser()

    file = Path(
        "/Users/arthurbrenno/Documents/Dev/Paragon/agentle/examples/curriculum.pdf"
    )

    if not file.exists():
        raise ValueError("File does not exist.")

    parsed_file = pdf_parser.parse(str(file))

    chunk_ids = qdrant.upsert_file(
        parsed_file, collection_name="test_collection", exists_behavior="ignore"
    )

    print(chunk_ids)

    # qdrant.delete_vectors(collection_name="test_collection", ids=chunk_ids)
