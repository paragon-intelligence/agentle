"""
In-memory session storage implementation.
"""

import asyncio
import fnmatch
import time
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import Any, override

from rsb.models.base_model import BaseModel

from agentle.sessions.session_store import SessionStore


class InMemorySessionStore[T_Session: BaseModel](SessionStore[T_Session]):
    """
    In-memory session storage implementation.

    This implementation stores sessions in memory with optional TTL support.
    Suitable for development, testing, and single-instance production deployments.

    Features:
    - Thread-safe operations
    - Automatic cleanup of expired sessions
    - Pattern-based session listing
    - Memory-efficient storage
    """

    def __init__(self, cleanup_interval_seconds: int = 300):
        """
        Initialize the in-memory session store.

        Args:
            cleanup_interval_seconds: Interval for automatic cleanup of expired sessions
        """
        self._sessions: MutableMapping[str, T_Session] = {}
        self._expiry_times: MutableMapping[str, float] = {}
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def _start_cleanup_task(self) -> None:
        """Start the automatic cleanup task if not already running."""
        if self._cleanup_task is None and not self._closed:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up expired sessions."""
        while not self._closed:
            try:
                await asyncio.sleep(self._cleanup_interval)
                if not self._closed:
                    await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue cleanup even if there's an error
                pass

    @override
    async def get_session(self, session_id: str) -> T_Session | None:
        """Retrieve a session by ID."""
        await self._start_cleanup_task()

        async with self._lock:
            # Check if session exists and is not expired
            if session_id in self._sessions:
                expiry_time = self._expiry_times.get(session_id)
                if expiry_time is None or time.time() < expiry_time:
                    return self._sessions[session_id]
                else:
                    # Session expired, remove it
                    del self._sessions[session_id]
                    del self._expiry_times[session_id]

            return None

    @override
    async def set_session(
        self, session_id: str, session: T_Session, ttl_seconds: int | None = None
    ) -> None:
        """Store a session."""
        await self._start_cleanup_task()

        async with self._lock:
            self._sessions[session_id] = session

            if ttl_seconds is not None:
                self._expiry_times[session_id] = time.time() + ttl_seconds
            elif session_id in self._expiry_times:
                # Remove expiry if TTL is None
                del self._expiry_times[session_id]

    @override
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                if session_id in self._expiry_times:
                    del self._expiry_times[session_id]
                return True
            return False

    @override
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists and is not expired."""
        session = await self.get_session(session_id)
        return session is not None

    @override
    async def list_sessions(self, pattern: str | None = None) -> Sequence[str]:
        """List all session IDs, optionally matching a pattern."""
        async with self._lock:
            current_time = time.time()

            # Filter out expired sessions
            active_sessions: MutableSequence[str] = []
            for session_id in self._sessions.keys():
                expiry_time = self._expiry_times.get(session_id)
                if expiry_time is None or current_time < expiry_time:
                    active_sessions.append(session_id)

            # Apply pattern filter if provided
            if pattern:
                active_sessions = [
                    session_id
                    for session_id in active_sessions
                    if fnmatch.fnmatch(session_id, pattern)
                ]

            return active_sessions

    @override
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        async with self._lock:
            current_time = time.time()
            expired_sessions: MutableSequence[str] = []

            for session_id, expiry_time in self._expiry_times.items():
                if current_time >= expiry_time:
                    expired_sessions.append(session_id)

            # Remove expired sessions
            for session_id in expired_sessions:
                del self._sessions[session_id]
                del self._expiry_times[session_id]

            return len(expired_sessions)

    @override
    async def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        # This will trigger cleanup of expired sessions
        active_sessions: Sequence[str] = await self.list_sessions()
        session_count: int = len(active_sessions)
        return session_count

    @override
    async def close(self) -> None:
        """Clean up resources and close connections."""
        self._closed = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        async with self._lock:
            self._sessions.clear()
            self._expiry_times.clear()

    def get_stats(self) -> Mapping[str, int]:
        """
        Get statistics about the session store.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_sessions": len(self._sessions),
            "sessions_with_ttl": len(self._expiry_times),
            "cleanup_interval_seconds": self._cleanup_interval,
        }
