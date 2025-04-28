import abc
from collections.abc import Sequence

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    Resource,
    TextResourceContents,
    Tool,
)
from rsb.models.base_model import BaseModel


class MCPServerProtocol(BaseModel, abc.ABC):
    """Base class for Model Context Protocol servers."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable name for the server."""
        ...

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """
        ...

    @abc.abstractmethod
    async def list_tools(self) -> Sequence[Tool]:
        """List the tools available on the server."""
        ...

    @abc.abstractmethod
    async def list_resources(self) -> Sequence[Resource]: ...

    @abc.abstractmethod
    async def list_resource_contents(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]: ...

    @abc.abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, object] | None
    ) -> CallToolResult:
        """Invoke a tool on the server."""
        ...
