"""
Session manager that coordinates session operations.
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from rsb.models.base_model import BaseModel

from agentle.sessions.session_store import SessionStore


class SessionManager[T_Session: BaseModel]:
    """
    Session manager that provides a high-level interface for session operations.

    This class coordinates between different session store implementations
    and provides additional features like session validation, event handling,
    and metadata management.
    """

    session_store: SessionStore[T_Session]
    default_ttl_seconds: int | None
    enable_events: bool
    _event_handlers: MutableMapping[str, list[Callable[..., Any]]]

    def __init__(
        self,
        session_store: SessionStore[T_Session],
        default_ttl_seconds: int | None = 3600,
        enable_events: bool = False,
    ):
        """
        Initialize the session manager.

        Args:
            session_store: The underlying session store implementation
            default_ttl_seconds: Default TTL for sessions
            enable_events: Whether to enable session events (created, updated, deleted)
        """
        self.session_store = session_store
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_events = enable_events
        self._event_handlers: MutableMapping[str, list[Callable[..., Any]]] = {
            "session_created": [],
            "session_updated": [],
            "session_deleted": [],
            "session_expired": [],
        }

    async def get_session(
        self,
        session_id: str,
        refresh_ttl: bool = False,
        additional_ttl_seconds: int | None = None,
    ) -> T_Session | None:
        """
        Get a session by ID with optional TTL refresh.

        Args:
            session_id: Session ID to retrieve
            refresh_ttl: Whether to refresh the TTL when accessing the session
            additional_ttl_seconds: Additional seconds to add to TTL if refreshing

        Returns:
            The session if found, None otherwise
        """
        session = await self.session_store.get_session(session_id)

        if session is not None and refresh_ttl:
            # Refresh TTL on access
            ttl = additional_ttl_seconds or self.default_ttl_seconds
            if ttl:
                await self.session_store.set_session(session_id, session, ttl)

        return session

    async def create_session(
        self,
        session_id: str,
        session: T_Session,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Create a new session.

        Args:
            session_id: Unique identifier for the session
            session: The session object to store
            ttl_seconds: TTL for the session (uses default if not provided)
            overwrite: Whether to overwrite if session already exists

        Returns:
            True if session was created, False if it already exists (and overwrite=False)
        """
        # Check if session already exists
        if not overwrite:
            existing = await self.session_store.exists(session_id)
            if existing:
                return False

        # Set TTL
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        # Store the session
        await self.session_store.set_session(session_id, session, ttl)

        # Fire event
        if self.enable_events:
            await self._fire_event("session_created", session_id, session)

        return True

    async def update_session(
        self,
        session_id: str,
        session: T_Session,
        ttl_seconds: int | None = None,
        create_if_missing: bool = False,
    ) -> bool:
        """
        Update an existing session.

        Args:
            session_id: Session ID to update
            session: Updated session object
            ttl_seconds: New TTL for the session
            create_if_missing: Whether to create the session if it doesn't exist

        Returns:
            True if session was updated, False if it doesn't exist (and create_if_missing=False)
        """
        # Check if session exists
        exists = await self.session_store.exists(session_id)
        if not exists and not create_if_missing:
            return False

        # Set TTL
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        # Store the session
        await self.session_store.set_session(session_id, session, ttl)

        # Fire event
        if self.enable_events:
            event_type = "session_created" if not exists else "session_updated"
            await self._fire_event(event_type, session_id, session)

        return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if it didn't exist
        """
        # Get session data before deletion for event
        session_data = None
        if self.enable_events:
            session_data = await self.session_store.get_session(session_id)

        # Delete the session
        deleted = await self.session_store.delete_session(session_id)

        # Fire event
        if deleted and self.enable_events:
            await self._fire_event("session_deleted", session_id, session_data)

        return deleted

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return await self.session_store.exists(session_id)

    async def list_sessions(
        self, pattern: str | None = None, include_metadata: bool = False
    ) -> Sequence[str] | list[Mapping[str, Any]]:
        """
        List session IDs or session metadata.

        Args:
            pattern: Optional pattern to filter session IDs
            include_metadata: Whether to include session metadata

        Returns:
            List of session IDs or session metadata dictionaries
        """
        session_ids = await self.session_store.list_sessions(pattern)

        if not include_metadata:
            return session_ids

        # Build metadata for each session
        sessions_with_metadata = []
        for session_id in session_ids:
            metadata: Mapping[str, Any] = {
                "session_id": session_id,
                "exists": True,  # We know it exists since we just listed it
            }

            # Add TTL information if available
            if hasattr(self.session_store, "get_session_ttl"):
                try:
                    ttl_method = getattr(self.session_store, "get_session_ttl")
                    ttl = await ttl_method(session_id)
                    metadata["ttl_seconds"] = ttl
                except Exception:
                    metadata["ttl_seconds"] = None

            sessions_with_metadata.append(metadata)

        return sessions_with_metadata

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        return await self.session_store.cleanup_expired()

    async def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        return await self.session_store.get_session_count()

    async def extend_session_ttl(
        self, session_id: str, additional_seconds: int
    ) -> bool:
        """
        Extend the TTL of a session.

        Args:
            session_id: Session ID to extend
            additional_seconds: Additional seconds to add to TTL

        Returns:
            True if TTL was extended, False if session doesn't exist
        """
        if hasattr(self.session_store, "extend_session_ttl"):
            extend_method = getattr(self.session_store, "extend_session_ttl")
            return await extend_method(session_id, additional_seconds)
        else:
            # Fallback: get session and re-store with extended TTL
            session = await self.session_store.get_session(session_id)
            if session is None:
                return False

            # Calculate new TTL (this is approximate since we don't know the current TTL)
            new_ttl = additional_seconds
            await self.session_store.set_session(session_id, session, new_ttl)
            return True

    def add_event_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """
        Add an event handler for session events.

        Args:
            event_type: Type of event (session_created, session_updated, session_deleted, session_expired)
            handler: Async callable to handle the event
        """
        if event_type not in self._event_handlers:
            raise ValueError(f"Unknown event type: {event_type}")

        self._event_handlers[event_type].append(handler)

    def remove_event_handler(
        self, event_type: str, handler: Callable[..., Any]
    ) -> bool:
        """
        Remove an event handler.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Returns:
            True if handler was removed, False if not found
        """
        if event_type not in self._event_handlers:
            return False

        try:
            self._event_handlers[event_type].remove(handler)
            return True
        except ValueError:
            return False

    async def _fire_event(
        self, event_type: str, session_id: str, session_data: T_Session | None
    ) -> None:
        """Fire an event to all registered handlers."""
        if not self.enable_events or event_type not in self._event_handlers:
            return

        handlers = self._event_handlers[event_type]
        for handler in handlers:
            try:
                # Try calling as async first, then sync
                try:
                    await handler(session_id, session_data)
                except TypeError:
                    # Might be a sync function
                    handler(session_id, session_data)
            except Exception:
                # Don't let event handler errors break session operations
                pass

    async def close(self) -> None:
        """Close the session manager and underlying store."""
        await self.session_store.close()

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get statistics about the session manager.

        Returns:
            Dictionary with statistics
        """
        handler_counts = {}
        for event_type, handlers in self._event_handlers.items():
            handler_counts[event_type] = len(handlers)

        base_stats = {
            "default_ttl_seconds": self.default_ttl_seconds,
            "events_enabled": self.enable_events,
            "event_handlers": handler_counts,
        }

        # Add store-specific stats if available
        if hasattr(self.session_store, "get_stats"):
            stats_method = getattr(self.session_store, "get_stats")
            base_stats["store_stats"] = stats_method()

        return base_stats
