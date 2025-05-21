"""
Redis-based implementation of the MCP session manager.

This module provides a Redis-backed session manager implementation
suitable for use in multi-process applications and production environments.
"""

import json
from typing import Dict, Optional, Any, cast

from agentle.mcp.session_management.session_manager import SessionManager


class RedisSessionManager(SessionManager):
    """
    Redis-backed implementation of the SessionManager interface.

    This implementation stores session data in Redis, making it suitable for
    use in multi-process applications and production environments where
    session data needs to be shared across multiple workers.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "mcp_session:",
        expiration_seconds: int = 3600,  # 1 hour default
    ) -> None:
        """
        Initialize the Redis session manager.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys to avoid collisions
            expiration_seconds: Time in seconds before sessions expire
        """
        import redis.asyncio as redis_async

        self._redis = redis_async.from_url(redis_url)
        self._key_prefix = key_prefix
        self._expiration_seconds = expiration_seconds

    def _get_redis_key(self, server_key: str) -> str:
        """
        Get the Redis key for a server key.

        Args:
            server_key: A unique identifier for the server connection

        Returns:
            str: The Redis key to use
        """
        return f"{self._key_prefix}{server_key}"

    async def get_session(self, server_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information for a specific server from Redis.

        Args:
            server_key: A unique identifier for the server connection

        Returns:
            Optional[Dict[str, Any]]: The session data if it exists, None otherwise
        """
        from redis.exceptions import RedisError

        try:
            redis_key = self._get_redis_key(server_key)
            data = await self._redis.get(redis_key)

            if data is None:
                return None

            return cast(Dict[str, Any], json.loads(data))
        except (RedisError, json.JSONDecodeError) as _:
            # Log error here if needed
            return None

    async def store_session(
        self, server_key: str, session_data: Dict[str, Any]
    ) -> None:
        """
        Store session information for a specific server in Redis.

        Args:
            server_key: A unique identifier for the server connection
            session_data: The session data to store
        """
        from redis.exceptions import RedisError

        try:
            redis_key = self._get_redis_key(server_key)
            serialized_data = json.dumps(session_data)

            await self._redis.set(
                redis_key, serialized_data, ex=self._expiration_seconds
            )
        except (RedisError, TypeError) as _:
            # Log error here if needed
            pass

    async def delete_session(self, server_key: str) -> None:
        """
        Delete session information for a specific server from Redis.

        Args:
            server_key: A unique identifier for the server connection
        """
        from redis.exceptions import RedisError

        try:
            redis_key = self._get_redis_key(server_key)
            await self._redis.delete(redis_key)
        except RedisError as _:
            # Log error here if needed
            pass

    async def close(self) -> None:
        """
        Close the Redis connection.
        """
        await self._redis.close()
