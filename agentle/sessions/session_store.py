"""
Base interface for session storage implementations.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rsb.models.base_model import BaseModel


@runtime_checkable
class SessionStore[T_Session: BaseModel](Protocol):
    """
    Abstract base class for session storage implementations.

    This interface defines the contract that all session stores must implement,
    enabling support for different storage backends (in-memory, Redis, database, etc.)
    """

    @abstractmethod
    async def get_session(self, session_id: str) -> T_Session | None:
        """
        Retrieve a session by ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            The session object if found, None otherwise
        """
        pass

    @abstractmethod
    async def set_session(
        self, session_id: str, session: T_Session, ttl_seconds: int | None = None
    ) -> None:
        """
        Store a session.

        Args:
            session_id: Unique identifier for the session
            session: The session object to store
            ttl_seconds: Optional time-to-live in seconds
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            True if the session was deleted, False if it didn't exist
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Unique identifier for the session

        Returns:
            True if the session exists, False otherwise
        """
        pass

    @abstractmethod
    async def list_sessions(self, pattern: str | None = None) -> Sequence[str]:
        """
        List all session IDs, optionally matching a pattern.

        Args:
            pattern: Optional pattern to filter session IDs

        Returns:
            List of session IDs
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        pass

    @abstractmethod
    async def get_session_count(self) -> int:
        """
        Get the total number of active sessions.

        Returns:
            Number of active sessions
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources and close connections."""
        pass
