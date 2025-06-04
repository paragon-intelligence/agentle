"""
Redis-based session storage implementation.
"""

from collections.abc import Mapping, Sequence
import json
from typing import override, Any, TYPE_CHECKING, cast
import fnmatch

from rsb.models.base_model import BaseModel

from agentle.sessions.session_store import SessionStore

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis

    redis_available = True
except ImportError:
    redis_available = False
    redis = None
    if TYPE_CHECKING:
        from redis.asyncio import Redis


class RedisSessionStore[T_Session: BaseModel](SessionStore[T_Session]):
    """
    Redis-based session storage implementation.

    This implementation stores sessions in Redis with automatic TTL support.
    Suitable for production deployments, especially in distributed environments.

    Features:
    - Distributed storage
    - Automatic expiration via Redis TTL
    - Pattern-based session listing
    - JSON serialization for session data
    - Connection pooling
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentle:session:",
        default_ttl_seconds: int | None = 3600,
        session_class: type[T_Session] | None = None,
    ):
        """
        Initialize the Redis session store.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all session keys in Redis
            default_ttl_seconds: Default TTL for sessions if not specified
            session_class: Class to deserialize sessions into
        """
        if not redis_available:
            raise ImportError(
                "Redis is required for RedisSessionStore. Install with: pip install redis"
            )

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds
        self.session_class = session_class
        self._redis: Redis | None = None

    async def _get_redis(self) -> "Redis":
        """Get or create Redis connection."""
        if self._redis is None:
            if redis is None:
                raise RuntimeError("Redis is not available")
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _make_key(self, session_id: str) -> str:
        """Create Redis key for session ID."""
        return f"{self.key_prefix}{session_id}"

    def _extract_session_id(self, redis_key: str) -> str:
        """Extract session ID from Redis key."""
        return redis_key[len(self.key_prefix) :]

    @override
    async def get_session(self, session_id: str) -> T_Session | None:
        """Retrieve a session by ID."""
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        session_data = await redis_client.get(key)
        if session_data is None:
            return None

        try:
            # Parse JSON data
            session_dict = json.loads(session_data)

            # If we have a session class, use it to deserialize
            if self.session_class:
                return self.session_class.model_validate(session_dict)
            else:
                # Return the raw dictionary if no class specified
                return session_dict  # type: ignore

        except (json.JSONDecodeError, ValueError):
            # If deserialization fails, delete the corrupted session
            await redis_client.delete(key)
            return None

    @override
    async def set_session(
        self, session_id: str, session: T_Session, ttl_seconds: int | None = None
    ) -> None:
        """Store a session."""
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        # Serialize session to JSON
        if isinstance(session, BaseModel):
            session_data = session.model_dump_json()
        else:
            session_data = json.dumps(session, default=str)

        # Determine TTL
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        if ttl is not None:
            await redis_client.setex(key, ttl, session_data)
        else:
            await redis_client.set(key, session_data)

    @override
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        result = await redis_client.delete(key)
        return result > 0

    @override
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        result = await redis_client.exists(key)
        return result > 0

    @override
    async def list_sessions(self, pattern: str | None = None) -> Sequence[str]:
        """List all session IDs, optionally matching a pattern."""
        redis_client = await self._get_redis()

        # Create Redis pattern for scanning keys
        if pattern:
            redis_pattern = f"{self.key_prefix}{pattern}"
        else:
            redis_pattern = f"{self.key_prefix}*"

        # Use SCAN for memory-efficient iteration
        session_ids: Sequence[str] = []
        async for redis_key in redis_client.scan_iter(match=redis_pattern):
            key_str = str(cast(str, redis_key)) if redis_key is not None else ""
            if key_str:
                session_id = self._extract_session_id(key_str)

                # Apply additional pattern filtering if needed
                if pattern is None or fnmatch.fnmatch(session_id, pattern):
                    session_ids.append(session_id)

        return session_ids

    @override
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Note: Redis handles TTL expiration automatically,
        so this method primarily serves to maintain interface compatibility.
        """
        # Redis automatically removes expired keys, so we don't need to do anything
        # We could optionally scan for keys that should have expired but haven't
        # due to Redis lazy expiration, but this is usually not necessary
        return 0

    @override
    async def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        session_ids = await self.list_sessions()
        return len(session_ids)

    @override
    async def close(self) -> None:
        """Clean up resources and close connections."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def get_session_ttl(self, session_id: str) -> int | None:
        """
        Get the remaining TTL for a session.

        Args:
            session_id: Session ID to check

        Returns:
            Remaining TTL in seconds, or None if no TTL or session doesn't exist
        """
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        ttl = await redis_client.ttl(key)
        if ttl == -2:  # Key doesn't exist
            return None
        elif ttl == -1:  # Key exists but has no TTL
            return None
        else:
            return ttl

    async def extend_session_ttl(
        self, session_id: str, additional_seconds: int
    ) -> bool:
        """
        Extend the TTL of an existing session.

        Args:
            session_id: Session ID to extend
            additional_seconds: Additional seconds to add to TTL

        Returns:
            True if TTL was extended, False if session doesn't exist
        """
        redis_client = await self._get_redis()
        key = self._make_key(session_id)

        # Get current TTL
        current_ttl = await redis_client.ttl(key)
        if current_ttl == -2:  # Key doesn't exist
            return False

        # Set new TTL
        if current_ttl == -1:  # Key has no TTL
            new_ttl = additional_seconds
        else:
            new_ttl = current_ttl + additional_seconds

        result = await redis_client.expire(key, new_ttl)
        return result

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get statistics about the session store.

        Returns:
            Dictionary with statistics
        """
        return {
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "default_ttl_seconds": self.default_ttl_seconds,
            "session_class": self.session_class.__name__
            if self.session_class
            else None,
        }
