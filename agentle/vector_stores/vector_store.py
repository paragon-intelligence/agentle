import abc
from collections.abc import MutableSequence, Sequence
import logging
from textwrap import dedent
from typing import Literal, ParamSpec

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embed_content import EmbedContent
from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.generations.tools.tool import Tool
from agentle.parsing.chunk import Chunk
from agentle.parsing.chunking.chunking_config import ChunkingConfig
from agentle.parsing.chunking.chunking_strategy import ChunkingStrategy
from agentle.parsing.parsed_file import ParsedFile
from agentle.vector_stores.collection import Collection
from agentle.vector_stores.create_collection_config import CreateCollectionConfig
from agentle.vector_stores.filters.field_condition import FieldCondition
from agentle.vector_stores.filters.filter import Filter
from agentle.vector_stores.filters.match_value import MatchValue

type ChunkID = str

logger = logging.getLogger(__name__)
SearchParams = ParamSpec("SearchParams")


class VectorStore(abc.ABC):
    default_collection_name: str
    embedding_provider: EmbeddingProvider
    generation_provider: GenerationProviderType | None
    detailed_agent_description: str | None
    """Use this to describe your vector store and tell the Agent when to use, what is this vector database etc why it is used etc."""

    def __init__(
        self,
        *,
        default_collection_name: str = "agentle",
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProviderType | None = None,
        detailed_agent_description: str | None = None,
    ) -> None:
        self.default_collection_name = default_collection_name
        self.embedding_provider = embedding_provider
        self.generation_provider = generation_provider
        self.detailed_agent_description = detailed_agent_description

    def find_related_content(
        self,
        query: str | Embedding | Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]:
        return run_sync(
            self.find_related_content_async,
            query=query,
            filter=filter,
            k=k,
            collection_name=collection_name,
        )

    async def find_related_content_async(
        self,
        query: str | Embedding | Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]:
        if query is None and filter is None:
            raise ValueError("Either query or filter must be provided.")

        if query:
            match query:
                case str():
                    embedding = await self.embedding_provider.generate_embeddings_async(
                        contents=query
                    )

                    return await self._find_related_content_async(
                        query=embedding.embeddings.value,
                        k=k,
                        filter=filter,
                        collection_name=collection_name,
                    )
                case Embedding():
                    return await self._find_related_content_async(
                        query=query.value,
                        k=k,
                        filter=filter,
                        collection_name=collection_name,
                    )
                case Sequence():
                    return await self._find_related_content_async(
                        query=query,
                        filter=filter,
                        k=k,
                        collection_name=collection_name,
                    )
        return await self._find_related_content_async(
            query=None, filter=filter, k=k, collection_name=collection_name
        )

    @abc.abstractmethod
    async def _find_related_content_async(
        self,
        query: Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]: ...

    def delete_vectors(
        self,
        collection_name: str,
        ids: Sequence[str] | None = None,
        filter: Filter | None = None,
    ) -> None:
        return run_sync(
            self.delete_vectors_async,
            collection_name=collection_name,
            ids=ids,
            filter=filter,
        )

    async def delete_vectors_async(
        self,
        collection_name: str,
        ids: Sequence[str] | None = None,
        filter: Filter | None = None,
    ) -> None:
        extra_ids: Sequence[Chunk] = (
            await self.find_related_content_async(filter=filter) if filter else []
        )

        _ids = list(list(ids or []) + list([c.id for c in extra_ids]))

        await self._delete_vectors_async(
            collection_name=collection_name, ids=list(set(_ids))
        )

    @abc.abstractmethod
    async def _delete_vectors_async(
        self,
        collection_name: str,
        ids: Sequence[str],
    ) -> None: ...

    def upsert(
        self,
        points: Embedding | Sequence[float],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
    ) -> None:
        return run_sync(
            self.upsert_async,
            points=points,
            timeout=timeout,
            collection_name=collection_name,
        )

    async def upsert_async(
        self,
        points: Embedding | Sequence[float],
        *,
        collection_name: str | None = None,
    ) -> None:
        if len(points) == 0:
            return None

        if isinstance(points, Sequence):
            return await self._upsert_async(
                points=Embedding(value=points),
                collection_name=collection_name,
            )

        return await self._upsert_async(
            points=points,
            collection_name=collection_name,
        )

    @abc.abstractmethod
    async def _upsert_async(
        self,
        points: Embedding,
        *,
        collection_name: str | None = None,
    ) -> None: ...

    def upsert_file(
        self,
        file: ParsedFile,
        *,
        timeout: float | None = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunking_config: ChunkingConfig | None = None,
        collection_name: str | None = None,
        exists_behavior: Literal["override", "error", "ignore"] = "error",
    ) -> Sequence[ChunkID]:
        return run_sync(
            self.upsert_file_async,
            file=file,
            timeout=timeout,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            collection_name=collection_name,
            exists_behavior=exists_behavior,
        )

    async def upsert_file_async(
        self,
        file: ParsedFile,
        *,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunking_config: ChunkingConfig | None = None,
        collection_name: str | None = None,
        exists_behavior: Literal["override", "error", "ignore"] = "error",
    ) -> Sequence[ChunkID]:
        # Check if file was already ingested in the database.
        possible_file_chunks = self.find_related_content(
            collection_name=collection_name or self.default_collection_name,
            filter=Filter(
                must=FieldCondition(
                    key="source_document_id", match=MatchValue(value=file.unique_id)
                )
            ),
        )

        file_exists = len(possible_file_chunks) > 0

        if file_exists:
            match exists_behavior:
                case "error":
                    logger.debug("File exists. Raising error.")
                    raise FileExistsError(
                        "The provided file already exists in the database"
                    )
                case "override":
                    logger.debug("Overriding existing file in Vector Store.")
                    delete_ids = [p.id for p in possible_file_chunks]
                    await self.delete_vectors_async(
                        collection_name=collection_name or self.default_collection_name,
                        ids=delete_ids,
                    )
                case "ignore":
                    logger.debug("Ignoring existing file in Vector Store.")
                    return [c.id for c in possible_file_chunks]

        chunks: Sequence[Chunk] = await file.chunkify_async(
            strategy=chunking_strategy, config=chunking_config
        )

        embed_contents: Sequence[EmbedContent] = [
            await self.embedding_provider.generate_embeddings_async(
                c.text, metadata=c.metadata, id=c.id
            )
            for c in chunks
        ]

        ids: MutableSequence[str] = []

        for e in embed_contents:
            await self.upsert_async(
                points=e.embeddings,
                collection_name=collection_name,
            )

            ids.append(e.embeddings.id)

        return ids

    def create_collection(
        self, collection_name: str, *, config: CreateCollectionConfig
    ) -> None:
        return run_sync(
            self.create_collection_async, collection_name=collection_name, config=config
        )

    @abc.abstractmethod
    async def create_collection_async(
        self, collection_name: str, *, config: CreateCollectionConfig
    ) -> None: ...

    def delete_collection(self, collection_name: str) -> None:
        return run_sync(self.delete_collection_async, collection_name=collection_name)

    @abc.abstractmethod
    async def delete_collection_async(self, collection_name: str) -> None: ...

    def list_collections(self) -> Sequence[Collection]:
        return run_sync(self.list_collections_async)

    @abc.abstractmethod
    async def list_collections_async(self) -> Sequence[Collection]: ...

    def as_search_tool(self) -> Tool[..., str]:
        async def retrieval_augmented_generation_search(
            query: str, top_k: int = 5
        ) -> str:
            related_chunks = await self.find_related_content_async(query=query, k=top_k)
            chunk_descriptions = [chunk.describe() for chunk in related_chunks]
            return dedent(f"""\
            <RelatedChunks>
            {"\n\n".join(chunk_descriptions)}
            </RelatedChunks>
            """)

        tool = Tool.from_callable(
            retrieval_augmented_generation_search,
            description=dedent("""\
            Searches a vector database for text chunks relevant to a query.

            This tool is essential for answering questions or finding information
            contained within a specific, indexed knowledge base. Use it when a user's
            query pertains to specific documents, files, or internal data that is
            not part of your general knowledge. It helps ground your answers in
            factual, source-based information.

            Args:
                query (str): The search query used to find relevant information. To
                             ensure the best results, formulate this query carefully:
                             - Be Specific and Detailed: Include proper nouns, technical
                               terms, dates, project codes, or any unique identifiers
                               from the user's request. This significantly helps
                               narrow the search.
                             - Use Full Questions or Phrases: Instead of just keywords
                               (e.g., "marketing budget"), formulate a complete
                               question or a descriptive phrase (e.g., "What was the
                               approved marketing budget for Q4 2025?"). This provides
                               richer semantic context for the vector search.
                             - Extract the Core Intent: Distill the user's request to
                               its essential informational need. Remove conversational
                               filler. For example, if the user asks, "Hey, can you
                               look up our new remote work policy for me?", the ideal
                               query is "new remote work policy".
                top_k (int, optional): The maximum number of relevant text chunks
                                       to return. Defaults to 5.

            Returns:
                str: A string containing the search results formatted within
                     <RelatedChunks> tags. Each chunk includes the text content
                     and metadata (like the source document), providing context
                     for the information found.

            Handling Search Results:
                - The search may sometimes return no results or content that is not
                  relevant to the user's query. Your primary responsibility is to
                  adhere to the user's instructions on how to proceed in this scenario.
                - If the user has not provided specific instructions, you should adopt
                  the following default behavior:
                  1.  Do NOT invent an answer.
                  2.  Clearly and politely state that you could not find a specific
                      answer in the available documents.
                  3.  You may suggest that the user try rephrasing the question.
                  Example Response: "I searched the documents but couldn't find a
                  specific answer regarding the 'Q3 innovation fund.' You might
                  try rephrasing the query, perhaps with a project name included."
            """)
            + dedent(f"""\
                      <VectorStoreCustomDescription>
                      {self.detailed_agent_description}
                      </VectorStoreCustomDescription>
                      """)
            if self.detailed_agent_description
            else "",
        )
        return tool
