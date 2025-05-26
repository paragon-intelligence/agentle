"""
In-memory cache store implementation for parsed documents.

This module provides a thread-safe, in-memory cache store that stores parsed documents
in the process memory with TTL support and automatic cleanup.
"""

import time
import threading
import weakref
from typing import Any, override

from rsb.models.field import Field

from agentle.parsing.cache.document_cache_store import DocumentCacheStore, CacheTTL
from agentle.parsing.parsed_document import ParsedDocument


class InMemoryDocumentCacheStore(DocumentCacheStore):
    """
    Thread-safe in-memory cache store for parsed documents.

    This cache store keeps parsed documents in memory with TTL support and automatic
    cleanup of expired entries. It's suitable for single-process applications and
    development environments.

    Features:
    - Thread-safe operations using locks
    - Automatic cleanup of expired entries
    - TTL support with "infinite" option
    - Memory-efficient with weak references for cleanup tracking

    Attributes:
        cleanup_interval: How often to run the cleanup timer in seconds

    Example:
        ```python
        from agentle.parsing.cache import InMemoryDocumentCacheStore

        # Create cache with 5-minute cleanup interval
        cache = InMemoryDocumentCacheStore(cleanup_interval=300)

        # Store a document with 1-hour TTL
        await cache.set_async("doc_key", parsed_doc, ttl=3600)

        # Retrieve the document
        cached_doc = await cache.get_async("doc_key")
        ```
    """

    cleanup_interval: int = Field(
        default=300,  # 5 minutes
        description="How often to run the cache cleanup timer in seconds",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize the cache store after Pydantic model creation."""
        # Cache storage: {key: (document, timestamp, ttl)}
        self._cache_store: dict[str, tuple[ParsedDocument, float, CacheTTL]] = {}
        self._cache_lock = threading.RLock()
        self._cleanup_timer: threading.Timer | None = None

        # Class-level tracking for resource management
        InMemoryDocumentCacheStore._instances.add(self)

        # Start cleanup timer if cleanup interval is set
        if self.cleanup_interval > 0:
            self._start_cleanup_timer()

    # Class-level tracking for proper cleanup
    _instances: weakref.WeakSet["InMemoryDocumentCacheStore"] = weakref.WeakSet()
    _class_lock = threading.RLock()

    @override
    async def get_async(self, key: str) -> ParsedDocument | None:
        """
        Retrieve a parsed document from the in-memory cache.

        This method checks if the document exists and hasn't expired before returning it.
        Expired documents are automatically removed from the cache.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached ParsedDocument if found and not expired, None otherwise
        """
        with self._cache_lock:
            if key not in self._cache_store:
                return None

            document, timestamp, ttl = self._cache_store[key]

            # Check if the entry has expired
            if self._is_expired(timestamp, ttl):
                # Remove expired entry
                del self._cache_store[key]
                return None

            return document

    @override
    async def set_async(
        self, key: str, value: ParsedDocument, ttl: CacheTTL = None
    ) -> None:
        """
        Store a parsed document in the in-memory cache.

        Args:
            key: The cache key to store under
            value: The ParsedDocument to cache
            ttl: Time to live for the cache entry
        """
        if ttl is None:
            # Don't store if no TTL is specified
            return

        with self._cache_lock:
            self._cache_store[key] = (value, time.time(), ttl)

    @override
    async def delete_async(self, key: str) -> bool:
        """
        Delete a cached document from memory.

        Args:
            key: The cache key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._cache_lock:
            if key in self._cache_store:
                del self._cache_store[key]
                return True
            return False

    @override
    async def clear_async(self) -> None:
        """
        Clear all cached documents from memory.
        """
        with self._cache_lock:
            self._cache_store.clear()

    @override
    async def exists_async(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists and is not expired, False otherwise
        """
        with self._cache_lock:
            if key not in self._cache_store:
                return False

            _, timestamp, ttl = self._cache_store[key]

            # Check if the entry has expired
            if self._is_expired(timestamp, ttl):
                # Remove expired entry
                del self._cache_store[key]
                return False

            return True

    def _is_expired(self, timestamp: float, ttl: CacheTTL) -> bool:
        """
        Check if a cache entry has expired.

        Args:
            timestamp: When the entry was stored
            ttl: The time-to-live setting

        Returns:
            True if the entry has expired, False otherwise
        """
        if ttl == "infinite":
            return False

        if ttl is None:
            return True  # No TTL means immediate expiry

        return time.time() - timestamp >= ttl

    def _start_cleanup_timer(self) -> None:
        """Start a timer that periodically cleans up expired cache entries."""
        with self._cache_lock:
            # Cancel any existing timer
            if self._cleanup_timer is not None:
                self._cleanup_timer.cancel()

            # Create a new timer
            self._cleanup_timer = threading.Timer(
                self.cleanup_interval, self._timer_callback
            )
            self._cleanup_timer.daemon = True  # Don't keep the application running
            self._cleanup_timer.start()

    def _timer_callback(self) -> None:
        """Callback function for the timer to clean cache and restart timer."""
        try:
            self._cleanup_expired_cache()
            # Restart the timer for continuous cleanup
            self._start_cleanup_timer()
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
            # Try to restart the timer even if cleanup failed
            self._start_cleanup_timer()

    def _cleanup_expired_cache(self) -> None:
        """
        Clean up expired cache entries in a thread-safe manner.
        """
        with self._cache_lock:
            # Find expired keys
            expired_keys = [
                key
                for key, (_, timestamp, ttl) in self._cache_store.items()
                if self._is_expired(timestamp, ttl)
            ]

            # Remove expired entries
            for key in expired_keys:
                del self._cache_store[key]

    def __del__(self):
        """Cleanup when the cache store is destroyed."""
        if hasattr(self, "_cleanup_timer") and self._cleanup_timer is not None:
            self._cleanup_timer.cancel()

    @classmethod
    def cleanup_all_instances(cls) -> None:
        """
        Clean up all active cache store instances.

        This method can be called during application shutdown to ensure
        all timers are properly cancelled.
        """
        with cls._class_lock:
            for instance in cls._instances:
                if (
                    hasattr(instance, "_cleanup_timer")
                    and instance._cleanup_timer is not None
                ):
                    instance._cleanup_timer.cancel()

    def get_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including total entries,
            expired entries, and memory usage information
        """
        with self._cache_lock:
            total_entries = len(self._cache_store)

            expired_count = sum(
                1
                for _, timestamp, ttl in self._cache_store.values()
                if self._is_expired(timestamp, ttl)
            )

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "active_entries": total_entries - expired_count,
            }
