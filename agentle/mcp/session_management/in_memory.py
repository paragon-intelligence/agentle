"""
In-memory implementation of the MCP session manager.

This module provides a thread-safe in-memory session manager implementation
suitable for use in single-process applications.
"""

import threading
from typing import Dict, Optional, Any

from agentle.mcp.session_management.session_manager import SessionManager


class InMemorySessionManager(SessionManager):
    """
    Thread-safe in-memory implementation of the SessionManager interface.

    This implementation stores session data in an in-memory dictionary and
    uses a threading lock to ensure thread safety. It is suitable for
    single-process applications but will not share session data across
    multiple processes or workers.
    """

    def __init__(self) -> None:
        """Initialize the in-memory session manager with an empty store."""
        self._session_store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def get_session(self, server_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection

        Returns:
            Optional[Dict[str, Any]]: The session data if it exists, None otherwise
        """
        with self._lock:
            return self._session_store.get(server_key, None)

    async def store_session(
        self, server_key: str, session_data: Dict[str, Any]
    ) -> None:
        """
        Store session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection
            session_data: The session data to store
        """
        with self._lock:
            self._session_store[server_key] = session_data

    async def delete_session(self, server_key: str) -> None:
        """
        Delete session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection
        """
        with self._lock:
            if server_key in self._session_store:
                del self._session_store[server_key]

    async def close(self) -> None:
        """
        Close any resources used by the session manager.

        For the in-memory implementation, this is a no-op as there are no
        external resources to close.
        """
        pass
