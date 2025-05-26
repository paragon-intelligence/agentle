"""
Abstract base class for cache stores used in document parsing.

This module defines the interface that all cache store implementations must follow
for caching parsed documents in the Agentle framework.
"""

import abc
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict

from agentle.parsing.parsed_document import ParsedDocument

type CacheTTL = int | Literal["infinite"] | None


class DocumentCacheStore(BaseModel, abc.ABC):
    """
    Abstract base class for cache stores used in document parsing.

    This class defines the interface that all cache store implementations must follow.
    Cache stores are responsible for storing and retrieving parsed documents to avoid
    redundant parsing operations.

    The cache store supports different TTL (Time To Live) strategies:
    - int value: Cache entries expire after this many seconds
    - "infinite": Cache entries never expire (until process restart or manual eviction)
    - None: No caching is performed

    Example:
        ```python
        # Using an in-memory cache store
        from agentle.parsing.cache import InMemoryDocumentCacheStore

        cache = InMemoryDocumentCacheStore()

        # Store a parsed document
        await cache.set_async("document_key", parsed_doc, ttl=3600)  # Cache for 1 hour

        # Retrieve a parsed document
        cached_doc = await cache.get_async("document_key")
        if cached_doc:
            print("Found cached document!")
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Remove frozen=True to allow post-init modifications
    )

    @abc.abstractmethod
    async def get_async(self, key: str) -> ParsedDocument | None:
        """
        Retrieve a parsed document from the cache asynchronously.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached ParsedDocument if found and not expired, None otherwise

        Example:
            ```python
            cached_doc = await cache.get_async("my_document_key")
            if cached_doc:
                print(f"Found cached document: {cached_doc.name}")
            else:
                print("Document not in cache or expired")
            ```
        """
        ...

    @abc.abstractmethod
    async def set_async(
        self, key: str, value: ParsedDocument, ttl: CacheTTL = None
    ) -> None:
        """
        Store a parsed document in the cache asynchronously.

        Args:
            key: The cache key to store under
            value: The ParsedDocument to cache
            ttl: Time to live for the cache entry:
                - int: Expire after this many seconds
                - "infinite": Never expire (until process restart)
                - None: Use default behavior (no caching)

        Example:
            ```python
            # Cache for 1 hour
            await cache.set_async("doc_key", parsed_doc, ttl=3600)

            # Cache indefinitely
            await cache.set_async("doc_key", parsed_doc, ttl="infinite")

            # No caching (immediate expiry)
            await cache.set_async("doc_key", parsed_doc, ttl=None)
            ```
        """
        ...

    @abc.abstractmethod
    async def delete_async(self, key: str) -> bool:
        """
        Delete a cached document asynchronously.

        Args:
            key: The cache key to delete

        Returns:
            True if the key was found and deleted, False otherwise

        Example:
            ```python
            deleted = await cache.delete_async("my_document_key")
            if deleted:
                print("Document removed from cache")
            else:
                print("Document was not in cache")
            ```
        """
        ...

    @abc.abstractmethod
    async def clear_async(self) -> None:
        """
        Clear all cached documents asynchronously.

        This method removes all entries from the cache store.

        Example:
            ```python
            await cache.clear_async()
            print("All cached documents cleared")
            ```
        """
        ...

    @abc.abstractmethod
    async def exists_async(self, key: str) -> bool:
        """
        Check if a key exists in the cache asynchronously.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists and is not expired, False otherwise

        Example:
            ```python
            if await cache.exists_async("my_document_key"):
                print("Document is cached")
            else:
                print("Document is not cached or expired")
            ```
        """
        ...

    def get_cache_key(self, content: str, parser_name: str = "default") -> str:
        """
        Generate a cache key for the given content and parser.

        This method creates a consistent cache key based on the content
        and parser used. The default implementation uses a hash of the content.

        Args:
            content: The content to generate a key for (file path, URL, or raw text)
            parser_name: The name of the parser being used

        Returns:
            A string cache key

        Example:
            ```python
            key = cache.get_cache_key("path/to/document.pdf", "pdf_parser")
            print(f"Cache key: {key}")
            ```
        """
        import hashlib

        # Create a hash of the content and parser name for consistent keys
        content_hash = hashlib.sha256(f"{content}:{parser_name}".encode()).hexdigest()
        return f"parsed_doc:{content_hash}"
