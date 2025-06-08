"""
In-memory cache store implementation for parsed documents.

This module provides a thread-safe, in-memory cache store that stores parsed documents
in the process memory with TTL support and automatic cleanup.
"""

import atexit
import logging
import threading
import time
import weakref
from typing import Optional, override

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.parsing.cache.document_cache_store import CacheTTL, DocumentCacheStore
from agentle.parsing.parsed_document import ParsedDocument

logger = logging.getLogger(__name__)


# Module-level tracking to avoid class attribute issues with Pydantic
_GLOBAL_INSTANCES: list[weakref.ReferenceType["InMemoryDocumentCacheStore"]] = []
_GLOBAL_LOCK = threading.RLock()
_CLEANUP_REGISTERED = threading.Event()


def _ensure_cleanup_registered() -> None:
    """Ensure that the module-level cleanup is registered with atexit."""
    if not _CLEANUP_REGISTERED.is_set():
        with _GLOBAL_LOCK:
            if not _CLEANUP_REGISTERED.is_set():
                atexit.register(_cleanup_all_instances)
                _CLEANUP_REGISTERED.set()


def _cleanup_all_instances() -> None:
    """Clean up all active cache store instances."""
    try:
        with _GLOBAL_LOCK:
            # Clean up dead references and get live instances
            instances_to_cleanup = []
            live_refs = []

            for ref in _GLOBAL_INSTANCES:
                instance = ref()
                if instance is not None:
                    instances_to_cleanup.append(instance)
                    live_refs.append(ref)

            # Update the global list with only live references
            _GLOBAL_INSTANCES[:] = live_refs

            # Cleanup live instances
            for instance in instances_to_cleanup:
                try:
                    instance.shutdown()
                except Exception as e:
                    logger.error(f"Error cleaning up cache instance: {e}")
    except Exception as e:
        logger.error(f"Error during global cleanup: {e}")


class InMemoryDocumentCacheStore(BaseModel, DocumentCacheStore):
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
    - Production-ready error handling and logging

    Attributes:
        cleanup_interval: How often to run the cleanup timer in seconds
        max_cache_size: Maximum number of entries in cache (0 = unlimited)

    Example:
        ```python
        from agentle.parsing.cache import InMemoryDocumentCacheStore

        # Create cache with 5-minute cleanup interval
        cache = InMemoryDocumentCacheStore(cleanup_interval=300, max_cache_size=1000)

        # Store a document with 1-hour TTL
        await cache.set_async("doc_key", parsed_doc, ttl=3600)

        # Retrieve the document
        cached_doc = await cache.get_async("doc_key")
        ```
    """

    cleanup_interval: int = Field(
        default=300,  # 5 minutes
        description="How often to run the cache cleanup timer in seconds",
        ge=0,
    )

    max_cache_size: int = Field(
        default=0,  # unlimited
        description="Maximum number of entries in cache (0 = unlimited)",
        ge=0,
    )

    model_config = ConfigDict(frozen=True)

    def __post_init__(self) -> None:
        """Initialize the InMemoryDocumentCacheStore."""

        # Initialize threading objects here to avoid Pydantic's deepcopy issues
        self._cache_store: dict[str, tuple[ParsedDocument, float, CacheTTL]] = {}
        self._cache_lock = threading.RLock()
        self._cleanup_timer: Optional[threading.Timer] = None
        self._is_shutdown = threading.Event()

        # Track this instance globally
        _ensure_cleanup_registered()
        with _GLOBAL_LOCK:
            _GLOBAL_INSTANCES.append(weakref.ref(self))

        # Start cleanup timer if cleanup interval is set
        if self.cleanup_interval > 0:
            self._start_cleanup_timer()

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
        if self._is_shutdown.is_set():
            return None

        try:
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
        except Exception as e:
            logger.error(f"Error retrieving cache key '{key}': {e}")
            return None

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
        if ttl is None or self._is_shutdown.is_set():
            # Don't store if no TTL is specified or if shutdown
            return

        try:
            with self._cache_lock:
                # Check cache size limit
                if (
                    self.max_cache_size > 0
                    and len(self._cache_store) >= self.max_cache_size
                ):
                    # Remove oldest entry if at limit
                    self._evict_oldest_entry()

                self._cache_store[key] = (value, time.time(), ttl)
        except Exception as e:
            logger.error(f"Error storing cache key '{key}': {e}")

    @override
    async def delete_async(self, key: str) -> bool:
        """
        Delete a cached document from memory.

        Args:
            key: The cache key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        if self._is_shutdown.is_set():
            return False

        try:
            with self._cache_lock:
                if key in self._cache_store:
                    del self._cache_store[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting cache key '{key}': {e}")
            return False

    @override
    async def clear_async(self) -> None:
        """
        Clear all cached documents from memory.
        """
        if self._is_shutdown.is_set():
            return

        try:
            with self._cache_lock:
                self._cache_store.clear()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    @override
    async def exists_async(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: The cache key to check

        Returns:
            True if the key exists and is not expired, False otherwise
        """
        if self._is_shutdown.is_set():
            return False

        try:
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
        except Exception as e:
            logger.error(f"Error checking cache key '{key}': {e}")
            return False

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

    def _evict_oldest_entry(self) -> None:
        """Evict the oldest cache entry to make room for new entries."""
        if not self._cache_store:
            return

        # Find the oldest entry by timestamp
        oldest_key = min(
            self._cache_store.keys(),
            key=lambda k: self._cache_store[k][1],  # timestamp is at index 1
        )
        del self._cache_store[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    def _start_cleanup_timer(self) -> None:
        """Start a timer that periodically cleans up expired cache entries."""
        if self._is_shutdown.is_set():
            return

        try:
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
        except Exception as e:
            logger.error(f"Error starting cleanup timer: {e}")

    def _timer_callback(self) -> None:
        """Callback function for the timer to clean cache and restart timer."""
        if self._is_shutdown.is_set():
            return

        try:
            self._cleanup_expired_cache()
            # Restart the timer for continuous cleanup
            if not self._is_shutdown.is_set():
                self._start_cleanup_timer()
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            # Try to restart the timer even if cleanup failed
            if not self._is_shutdown.is_set():
                self._start_cleanup_timer()

    def _cleanup_expired_cache(self) -> None:
        """
        Clean up expired cache entries in a thread-safe manner.
        """
        if self._is_shutdown.is_set():
            return

        try:
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

                if expired_keys:
                    logger.debug(
                        f"Cleaned up {len(expired_keys)} expired cache entries"
                    )
        except Exception as e:
            logger.error(f"Error during expired cache cleanup: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the cache store and cleanup resources.

        This method should be called when the cache store is no longer needed
        to ensure proper cleanup of timers and resources.
        """
        self._is_shutdown.set()

        try:
            if hasattr(self, "_cleanup_timer") and self._cleanup_timer is not None:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None

            # Clear the cache
            if hasattr(self, "_cache_lock"):
                with self._cache_lock:
                    self._cache_store.clear()

        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")

    def __del__(self):
        """Cleanup when the cache store is destroyed."""
        try:
            self.shutdown()
        except Exception:
            # Ignore errors during destruction
            pass

    @classmethod
    def cleanup_all_instances(cls) -> None:
        """
        Clean up all active cache store instances.

        This method can be called during application shutdown to ensure
        all timers are properly cancelled.
        """
        _cleanup_all_instances()

    def get_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including total entries,
            expired entries, and memory usage information
        """
        if self._is_shutdown.is_set():
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "active_entries": 0,
            }

        try:
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
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "total_entries": 0,
                "expired_entries": 0,
                "active_entries": 0,
            }
