"""
Redis cache store implementation for parsed documents.

This module provides a Redis-based cache store that allows for distributed caching
of parsed documents across multiple processes and servers.
"""

import json
from typing import Any, override

from rsb.models.field import Field

from agentle.parsing.cache.document_cache_store import DocumentCacheStore, CacheTTL
from agentle.parsing.parsed_document import ParsedDocument


class RedisCacheStore(DocumentCacheStore):
    """
    Redis-based cache store for parsed documents.

    This cache store uses Redis for distributed caching, making it suitable for
    production environments with multiple processes or servers. It supports
    TTL-based expiration and automatic serialization/deserialization of documents.

    Attributes:
        redis_url: Redis connection URL
        key_prefix: Prefix for all cache keys to avoid collisions
        default_ttl: Default TTL in seconds for cache entries

    Example:
        ```python
        from agentle.parsing.cache import RedisCacheStore

        # Create Redis cache store
        cache = RedisCacheStore(
            redis_url="redis://localhost:6379/0",
            key_prefix="agentle:parsed:",
            default_ttl=3600
        )

        # Store a document with custom TTL
        await cache.set_async("doc_key", parsed_doc, ttl=7200)  # Cache for 2 hours

        # Retrieve the document
        cached_doc = await cache.get_async("doc_key")
        ```
    """

    redis_url: str = Field(
        description="Redis connection URL (e.g., 'redis://localhost:6379/0')"
    )

    key_prefix: str = Field(
        default="agentle:parsed:",
        description="Prefix for all cache keys to avoid collisions",
    )

    default_ttl: int = Field(
        default=3600,  # 1 hour
        description="Default TTL in seconds for cache entries",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Redis cache store after Pydantic model creation."""
        # Import redis only when needed
        try:
            import redis.asyncio as redis

            self._redis_module = redis
            self._redis_client: redis.Redis | None = None
        except ImportError:
            raise ImportError(
                "Redis is required for RedisCacheStore. Install it with: pip install redis"
            )

    async def _get_redis_client(self):
        """Get or create Redis client connection."""
        if self._redis_client is None:
            self._redis_client = self._redis_module.from_url(self.redis_url)
        return self._redis_client

    def _get_full_key(self, key: str) -> str:
        """Get the full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    @override
    async def get_async(self, key: str) -> ParsedDocument | None:
        """
        Retrieve a parsed document from Redis cache.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached ParsedDocument if found and not expired, None otherwise
        """
        redis_client = await self._get_redis_client()
        full_key = self._get_full_key(key)

        try:
            # Get the serialized document from Redis
            serialized_data = await redis_client.get(full_key)

            if serialized_data is None:
                return None

            # Deserialize the document
            document_data = json.loads(serialized_data)

            # Reconstruct the ParsedDocument
            return ParsedDocument.model_validate(document_data)

        except Exception:
            # If there's any error (JSON parsing, validation, etc.), return None
            return None

    @override
    async def set_async(
        self, key: str, value: ParsedDocument, ttl: CacheTTL = None
    ) -> None:
        """
        Store a parsed document in Redis cache.

        Args:
            key: The cache key to store under
            value: The ParsedDocument to cache
            ttl: Time to live for the cache entry
        """
        if ttl is None:
            # Don't store if no TTL is specified
            return

        redis_client = await self._get_redis_client()
        full_key = self._get_full_key(key)

        try:
            # Serialize the document
            serialized_data = json.dumps(value.model_dump())

            # Determine the TTL to use
            if ttl == "infinite":
                # Set without expiration
                await redis_client.set(full_key, serialized_data)
            else:
                # Set with TTL (ttl is guaranteed to be int here)
                await redis_client.setex(full_key, ttl, serialized_data)

        except Exception:
            # Silently fail if we can't store in cache
            pass

    @override
    async def delete_async(self, key: str) -> bool:
        """
        Delete a cached document from Redis.

        Args:
            key: The cache key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        redis_client = await self._get_redis_client()
        full_key = self._get_full_key(key)

        try:
            result = await redis_client.delete(full_key)
            return result > 0
        except Exception:
            return False

    @override
    async def clear_async(self) -> None:
        """
        Clear all cached documents with the configured prefix from Redis.
        """
        redis_client = await self._get_redis_client()

        try:
            # Find all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys: list[Any] = []

            # Use scan_iter to handle large numbers of keys efficiently
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)

            # Delete all found keys
            if keys:
                await redis_client.delete(*keys)

        except Exception:
            # Silently fail if we can't clear the cache
            pass

    @override
    async def exists_async(self, key: str) -> bool:
        """
        Check if a key exists in Redis cache.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists, False otherwise
        """
        redis_client = await self._get_redis_client()
        full_key = self._get_full_key(key)

        try:
            result = await redis_client.exists(full_key)
            return result > 0
        except Exception:
            return False

    async def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the Redis cache.

        Returns:
            Dictionary with cache information including Redis info and key counts
        """
        redis_client = await self._get_redis_client()

        try:
            # Get Redis info
            redis_info = await redis_client.info()

            # Count keys with our prefix
            pattern = f"{self.key_prefix}*"
            key_count = 0
            async for _ in redis_client.scan_iter(match=pattern):
                key_count += 1

            return {
                "redis_version": redis_info.get("redis_version", "unknown"),
                "used_memory": redis_info.get("used_memory_human", "unknown"),
                "connected_clients": redis_info.get("connected_clients", 0),
                "total_keys_with_prefix": key_count,
                "key_prefix": self.key_prefix,
            }

        except Exception as e:
            return {
                "error": str(e),
                "key_prefix": self.key_prefix,
            }

    async def close(self) -> None:
        """
        Close the Redis connection.

        This method should be called when the cache store is no longer needed
        to properly clean up the Redis connection.
        """
        if self._redis_client is not None:
            await self._redis_client.close()
            self._redis_client = None
