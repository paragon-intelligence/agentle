"""
Suspension manager for handling suspended agent executions.

This module provides the SuspensionManager class that handles storing,
retrieving, and managing suspended agent contexts for Human-in-the-Loop
workflows where execution needs to be paused for external input.
"""

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict

from agentle.agents.context import Context


class SuspensionStore(ABC):
    """
    Abstract base class for storing suspended execution contexts.

    Implementations can use different storage backends like Redis,
    database, file system, etc.
    """

    @abstractmethod
    async def store_context(
        self,
        token: str,
        context: Context,
        metadata: Dict[str, Any],
        expires_at: datetime | None = None,
    ) -> None:
        """Store a suspended context with the given token."""
        pass

    @abstractmethod
    async def retrieve_context(
        self, token: str
    ) -> tuple[Context, Dict[str, Any]] | None:
        """Retrieve a suspended context by token. Returns None if not found."""
        pass

    @abstractmethod
    async def delete_context(self, token: str) -> bool:
        """Delete a suspended context. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    async def list_pending_approvals(
        self, user_id: str | None = None
    ) -> list[Dict[str, Any]]:
        """List pending approval requests, optionally filtered by user."""
        pass


class InMemorySuspensionStore(SuspensionStore):
    """
    In-memory implementation of SuspensionStore for development and testing.

    Note: This implementation loses data when the process restarts.
    For production, use a persistent store like Redis or database.
    """

    def __init__(self):
        self._store: Dict[str, tuple[Context, Dict[str, Any], datetime | None]] = {}

    async def store_context(
        self,
        token: str,
        context: Context,
        metadata: Dict[str, Any],
        expires_at: datetime | None = None,
    ) -> None:
        """Store a suspended context in memory."""
        self._store[token] = (context, metadata, expires_at)

    async def retrieve_context(
        self, token: str
    ) -> tuple[Context, Dict[str, Any]] | None:
        """Retrieve a suspended context from memory."""
        if token not in self._store:
            return None

        context, metadata, expires_at = self._store[token]

        # Check if expired
        if expires_at and datetime.now() > expires_at:
            del self._store[token]
            return None

        return context, metadata

    async def delete_context(self, token: str) -> bool:
        """Delete a suspended context from memory."""
        if token in self._store:
            del self._store[token]
            return True
        return False

    async def list_pending_approvals(
        self, user_id: str | None = None
    ) -> list[Dict[str, Any]]:
        """List pending approval requests from memory."""
        approvals = []

        for token, (context, metadata, expires_at) in self._store.items():
            # Skip expired entries
            if expires_at and datetime.now() > expires_at:
                continue

            # Filter by user if specified
            if user_id and metadata.get("user_id") != user_id:
                continue

            approval_info = {
                "token": token,
                "context_id": context.context_id,
                "reason": metadata.get("reason", "Unknown"),
                "created_at": metadata.get("created_at"),
                "expires_at": expires_at,
                "user_id": metadata.get("user_id"),
                "approval_data": metadata.get("approval_data", {}),
            }
            approvals.append(approval_info)

        return approvals


class SQLiteSuspensionStore(SuspensionStore):
    """
    SQLite-based implementation of SuspensionStore for persistent storage.

    This implementation provides persistence across process restarts
    and is suitable for single-instance deployments or development.
    """

    def __init__(self, db_path: str = "suspension_store.db"):
        """
        Initialize SQLite suspension store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suspended_contexts (
                    token TEXT PRIMARY KEY,
                    context_data TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    user_id TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON suspended_contexts(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON suspended_contexts(expires_at)
            """)
            conn.commit()

    async def store_context(
        self,
        token: str,
        context: Context,
        metadata: Dict[str, Any],
        expires_at: datetime | None = None,
    ) -> None:
        """Store a suspended context in SQLite."""
        # Serialize context and metadata
        context_data = context.model_dump_json()
        metadata_json = json.dumps(metadata)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO suspended_contexts 
                (token, context_data, metadata, created_at, expires_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    token,
                    context_data,
                    metadata_json,
                    datetime.now(),
                    expires_at,
                    metadata.get("user_id"),
                ),
            )
            conn.commit()

    async def retrieve_context(
        self, token: str
    ) -> tuple[Context, Dict[str, Any]] | None:
        """Retrieve a suspended context from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT context_data, metadata, expires_at 
                FROM suspended_contexts 
                WHERE token = ?
            """,
                (token,),
            )

            row = cursor.fetchone()
            if row is None:
                return None

            # Check if expired
            if row["expires_at"]:
                expires_at = datetime.fromisoformat(row["expires_at"])
                if datetime.now() > expires_at:
                    # Delete expired entry
                    conn.execute(
                        "DELETE FROM suspended_contexts WHERE token = ?", (token,)
                    )
                    conn.commit()
                    return None

            # Deserialize context and metadata
            context = Context.model_validate_json(row["context_data"])
            metadata = json.loads(row["metadata"])

            return context, metadata

    async def delete_context(self, token: str) -> bool:
        """Delete a suspended context from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM suspended_contexts WHERE token = ?", (token,)
            )
            conn.commit()
            return cursor.rowcount > 0

    async def list_pending_approvals(
        self, user_id: str | None = None
    ) -> list[Dict[str, Any]]:
        """List pending approval requests from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Clean up expired entries first
            conn.execute(
                """
                DELETE FROM suspended_contexts 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
                (datetime.now(),),
            )

            # Build query with optional user filter
            if user_id:
                query = """
                    SELECT token, context_data, metadata, created_at, expires_at, user_id
                    FROM suspended_contexts
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """
                params = [user_id]
            else:
                query = """
                    SELECT token, context_data, metadata, created_at, expires_at, user_id
                    FROM suspended_contexts
                    ORDER BY created_at DESC
                """
                params = []

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            approvals = []
            for row in rows:
                metadata = json.loads(row["metadata"])
                context_data = json.loads(row["context_data"])

                approval_info = {
                    "token": row["token"],
                    "context_id": context_data.get("context_id"),
                    "reason": metadata.get("reason", "Unknown"),
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "user_id": row["user_id"],
                    "approval_data": metadata.get("approval_data", {}),
                }
                approvals.append(approval_info)

            conn.commit()
            return approvals


class RedisSuspensionStore(SuspensionStore):
    """
    Redis-based implementation of SuspensionStore for distributed systems.

    This implementation provides persistence, scalability, and is suitable
    for production deployments with multiple instances.

    Requires: pip install redis
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentle:suspension:",
    ):
        """
        Initialize Redis suspension store.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
        """
        try:
            import redis.asyncio as redis  # type: ignore
        except ImportError:
            raise ImportError(
                "Redis is required for RedisSuspensionStore. "
                + "Install with: pip install redis"
            )

        self.redis = redis.from_url(redis_url)  # type: ignore
        self.key_prefix = key_prefix

    def _make_key(self, token: str) -> str:
        """Create a Redis key for the given token."""
        return f"{self.key_prefix}{token}"

    def _make_index_key(self, user_id: str | None = None) -> str:
        """Create a Redis key for the user index."""
        if user_id:
            return f"{self.key_prefix}user:{user_id}"
        return f"{self.key_prefix}all"

    async def store_context(
        self,
        token: str,
        context: Context,
        metadata: Dict[str, Any],
        expires_at: datetime | None = None,
    ) -> None:
        """Store a suspended context in Redis."""
        key = self._make_key(token)

        # Prepare data for storage
        data = {
            "context": context.model_dump_json(),
            "metadata": json.dumps(metadata),
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "user_id": metadata.get("user_id"),
        }

        # Store the context data
        await self.redis.hset(key, mapping=data)  # type: ignore

        # Set expiration if specified
        if expires_at:
            await self.redis.expireat(key, expires_at)  # type: ignore

        # Add to user index
        user_id = metadata.get("user_id")
        if user_id:
            await self.redis.sadd(self._make_index_key(user_id), token)  # type: ignore

        # Add to global index
        await self.redis.sadd(self._make_index_key(), token)  # type: ignore

    async def retrieve_context(
        self, token: str
    ) -> tuple[Context, Dict[str, Any]] | None:
        """Retrieve a suspended context from Redis."""
        key = self._make_key(token)

        # Get the data
        data = await self.redis.hgetall(key)  # type: ignore
        if not data:
            return None

            # Check if expired (Redis should handle this, but double-check)
        expires_at_bytes = data.get(b"expires_at")  # type: ignore
        if expires_at_bytes:
            expires_at_str = (
                expires_at_bytes.decode()  # type: ignore
                if hasattr(expires_at_bytes, "decode")  # type: ignore
                else str(expires_at_bytes)  # type: ignore
            )
            expires_at = datetime.fromisoformat(expires_at_str)  # type: ignore
            if datetime.now() > expires_at:
                await self.delete_context(token)
                return None

        # Deserialize context and metadata
        context_bytes = data.get(b"context")  # type: ignore
        metadata_bytes = data.get(b"metadata")  # type: ignore
        if not context_bytes or not metadata_bytes:
            return None

        context_str = (
            context_bytes.decode()  # type: ignore
            if hasattr(context_bytes, "decode")  # type: ignore
            else str(context_bytes)  # type: ignore
        )
        metadata_str = (
            metadata_bytes.decode()  # type: ignore
            if hasattr(metadata_bytes, "decode")  # type: ignore
            else str(metadata_bytes)  # type: ignore
        )

        context = Context.model_validate_json(context_str)  # type: ignore
        metadata = json.loads(metadata_str)  # type: ignore

        return context, metadata

    async def delete_context(self, token: str) -> bool:
        """Delete a suspended context from Redis."""
        key = self._make_key(token)

        # Get user_id before deletion for index cleanup
        user_id_bytes = await self.redis.hget(key, "user_id")  # type: ignore
        user_id = (
            user_id_bytes.decode()  # type: ignore
            if user_id_bytes and hasattr(user_id_bytes, "decode")  # type: ignore
            else None
        )

        # Delete the context
        deleted = await self.redis.delete(key)  # type: ignore

        if deleted:
            # Remove from indexes
            if user_id:
                await self.redis.srem(self._make_index_key(user_id), token)  # type: ignore
            await self.redis.srem(self._make_index_key(), token)  # type: ignore

        return deleted > 0

    async def list_pending_approvals(
        self, user_id: str | None = None
    ) -> list[Dict[str, Any]]:
        """List pending approval requests from Redis."""
        # Get tokens from appropriate index
        index_key = self._make_index_key(user_id)
        tokens = await self.redis.smembers(index_key)  # type: ignore

        approvals = []

        for token_bytes in tokens:  # type: ignore
            token = token_bytes.decode()  # type: ignore
            key = self._make_key(token)  # type: ignore

            # Get context data
            data = await self.redis.hgetall(key)  # type: ignore
            if not data:
                # Clean up stale index entry
                await self.redis.srem(index_key, token)  # type: ignore
                continue

            # Parse data
            try:
                metadata = json.loads(data["metadata"].decode())  # type: ignore
                context_data = json.loads(data["context"].decode())  # type: ignore

                approval_info = {
                    "token": token,
                    "context_id": context_data.get("context_id"),  # type: ignore
                    "reason": metadata.get("reason", "Unknown"),  # type: ignore
                    "created_at": data["created_at"].decode(),  # type: ignore
                    "expires_at": data["expires_at"].decode()  # type: ignore
                    if data.get("expires_at")
                    else None,
                    "user_id": data["user_id"].decode()  # type: ignore
                    if data.get("user_id")
                    else None,
                    "approval_data": metadata.get("approval_data", {}),  # type: ignore
                }
                approvals.append(approval_info)
            except (json.JSONDecodeError, KeyError):
                # Clean up corrupted entry
                await self.delete_context(token)  # type: ignore
                continue

        # Sort by creation time (newest first)
        approvals.sort(key=lambda x: x["created_at"], reverse=True)  # type: ignore
        return approvals


class SuspensionManager:
    """
    Manages suspended agent executions for Human-in-the-Loop workflows.

    This class handles the storage, retrieval, and lifecycle management
    of suspended agent contexts that are waiting for external input.
    """

    def __init__(self, store: SuspensionStore | None = None):
        """
        Initialize the suspension manager.

        Args:
            store: Storage backend for suspended contexts.
                  Defaults to InMemorySuspensionStore for development.
        """
        self.store = store or InMemorySuspensionStore()

    async def suspend_execution(
        self,
        context: Context,
        reason: str,
        approval_data: Dict[str, Any] | None = None,
        user_id: str | None = None,
        timeout_hours: int = 24,
    ) -> str:
        """
        Suspend an agent execution and return a resumption token.

        Args:
            context: The agent context to suspend
            reason: Human-readable reason for suspension
            approval_data: Data needed for the approval process
            user_id: ID of the user who can approve this request
            timeout_hours: Hours until the suspension expires

        Returns:
            A unique token that can be used to resume execution
        """
        # Generate unique resumption token
        token = str(uuid.uuid4())

        # Pause the context
        context.pause_execution(reason)

        # Calculate expiration
        expires_at = datetime.now() + timedelta(hours=timeout_hours)

        # Prepare metadata
        metadata = {
            "reason": reason,
            "approval_data": approval_data or {},
            "user_id": user_id,
            "created_at": datetime.now(),
            "timeout_hours": timeout_hours,
        }

        # Store the suspended context
        await self.store.store_context(token, context, metadata, expires_at)

        return token

    async def resume_execution(
        self, token: str, approval_result: Dict[str, Any] | None = None
    ) -> tuple[Context, Dict[str, Any]] | None:
        """
        Resume a suspended execution with approval result.

        Args:
            token: The resumption token
            approval_result: Result of the approval process

        Returns:
            Tuple of (context, metadata) if found, None otherwise
        """
        # Retrieve the suspended context
        result = await self.store.retrieve_context(token)
        if result is None:
            return None

        context, metadata = result

        # Add approval result to context checkpoint data
        if approval_result:
            context.set_checkpoint_data("approval_result", approval_result)

        # Resume the context
        context.resume_execution()

        # Clean up the stored context
        await self.store.delete_context(token)

        return context, metadata

    async def get_pending_approvals(
        self, user_id: str | None = None
    ) -> list[Dict[str, Any]]:
        """
        Get list of pending approval requests.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of pending approval information
        """
        return await self.store.list_pending_approvals(user_id)

    async def approve_request(
        self,
        token: str,
        approved: bool,
        approver_id: str,
        approval_data: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Approve or deny a suspended request.

        Args:
            token: The resumption token
            approved: Whether the request is approved
            approver_id: ID of the person making the approval decision
            approval_data: Additional approval data

        Returns:
            True if the approval was processed, False if token not found
        """
        result = await self.store.retrieve_context(token)
        if result is None:
            return False

        context, metadata = result

        # Prepare approval result
        approval_result = {
            "approved": approved,
            "approver_id": approver_id,
            "approved_at": datetime.now(),
            "approval_data": approval_data or {},
        }

        # Add to context
        context.set_checkpoint_data("approval_result", approval_result)

        if approved:
            # Resume execution
            context.resume_execution()
        else:
            # Mark as failed
            context.fail_execution(f"Request denied by {approver_id}")

        # Update the stored context
        await self.store.store_context(token, context, metadata)

        return True

    async def cleanup_expired(self) -> int:
        """
        Clean up expired suspension requests.

        Returns:
            Number of expired requests cleaned up
        """
        # This would be implemented by the specific store
        # For now, return 0 as the stores handle expiration internally
        return 0


# Global default suspension manager instance
_default_manager: SuspensionManager | None = None


def get_default_suspension_manager() -> SuspensionManager:
    """Get the default global suspension manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SuspensionManager()
    return _default_manager


def set_default_suspension_manager(manager: SuspensionManager) -> None:
    """Set the default global suspension manager instance."""
    global _default_manager
    _default_manager = manager
