"""
Abstract base class defining the interface for MCP session management.

This module provides the interface that all session management implementations must follow,
ensuring consistent behavior across different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class SessionManager(ABC):
    """
    Abstract base class for MCP session management.

    This class defines the interface that all session management implementations
    must implement to handle MCP session data across requests and potentially
    across different processes.
    """

    @abstractmethod
    async def get_session(self, server_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection

        Returns:
            Optional[Dict[str, Any]]: The session data if it exists, None otherwise
        """
        pass

    @abstractmethod
    async def store_session(
        self, server_key: str, session_data: Dict[str, Any]
    ) -> None:
        """
        Store session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection
            session_data: The session data to store
        """
        pass

    @abstractmethod
    async def delete_session(self, server_key: str) -> None:
        """
        Delete session information for a specific server.

        Args:
            server_key: A unique identifier for the server connection
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close any resources used by the session manager.
        """
        pass
