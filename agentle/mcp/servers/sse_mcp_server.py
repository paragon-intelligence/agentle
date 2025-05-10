"""
HTTP implementation of the Model Control Protocol (MCP) server.

This module provides an HTTP client implementation for interacting with MCP servers.
It enables connection management, tool discovery, resource querying, and tool execution
through standard HTTP endpoints.

The implementation follows the MCPServerProtocol interface and uses httpx for
asynchronous HTTP communication.
"""

import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from rsb.models.any_url import AnyUrl
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

if TYPE_CHECKING:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        Resource,
        TextResourceContents,
        Tool,
    )


class SSEMCPServer(MCPServerProtocol):
    """
    HTTP implementation of the MCP (Model Control Protocol) server.

    This class provides a client implementation for interacting with remote MCP servers
    over HTTP. It handles connection management, tool discovery, resource management,
    and tool invocation through HTTP endpoints.

    Attributes:
        server_name (str): A human-readable name for the server
        server_url (AnyUrl): The base URL of the HTTP server
        headers (dict[str, str]): HTTP headers to include with each request
        timeout_in_seconds (float): Request timeout in seconds

    Usage:
        server = SSEMCPServer(server_name="Example MCP", server_url="http://example.com/api")
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("tool_name", {"param": "value"})
        await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    server_url: AnyUrl = Field(..., description="Base URL for the HTTP MCP server")

    # Optional configuration fields
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to include with each request",
    )
    timeout_in_seconds: float = Field(
        default=100.0, description="Timeout in seconds for HTTP requests"
    )

    # Internal state
    _client: httpx.AsyncClient | None = None
    # _logger: logging.Logger = Field(
    #     default_factory=lambda: logging.getLogger(__name__),
    #     description="Logger instance for this class",
    # )
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )

    async def connect(self) -> None:
        """
        Connect to the HTTP MCP server.

        Establishes an HTTP client connection to the server and verifies connectivity
        by performing a test request to the root endpoint. The server is expected to
        remain connected until `cleanup()` is called.

        Raises:
            ConnectionError: If the connection to the server cannot be established
        """
        self._logger.info(f"Conectando ao servidor HTTP: {self.server_url}")
        self._client = httpx.AsyncClient(base_url=str(self.server_url), timeout=30.0)

        # Verificar conexão com o servidor
        try:
            response = await self._client.get("/")
            if response.status_code != 200:
                self._logger.warning(
                    f"Servidor respondeu com status {response.status_code}"
                )
        except Exception as e:
            self._logger.error(f"Erro ao conectar com servidor: {e}")
            await self.cleanup()
            raise ConnectionError(
                f"Não foi possível conectar ao servidor {self.server_url}: {e}"
            )

    @property
    def name(self) -> str:
        """
        Get a readable name for the server.

        Returns:
            str: The human-readable server name
        """
        return self.server_name

    async def cleanup(self) -> None:
        """
        Cleanup the server connection.

        Closes the HTTP client connection if it exists. This method should be called
        when the server connection is no longer needed.
        """
        if self._client is not None:
            self._logger.info(f"Fechando conexão com servidor HTTP: {self.server_url}")
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def ensure_connection(self) -> AsyncGenerator[None, None]:
        """
        Context manager to ensure connection is established before operations.

        This context manager wraps HTTP operations to provide consistent error handling
        for connection-related issues.

        Raises:
            httpx.RequestError: If there's an error during the HTTP request
        """
        try:
            yield
        except httpx.RequestError as e:
            self._logger.error(f"Erro na requisição HTTP: {e}")
            raise

    async def list_tools(self) -> Sequence["Tool"]:
        """
        List the tools available on the server.

        Retrieves the list of available tools from the /tools endpoint.

        Returns:
            Sequence[Tool]: A list of Tool objects available on the server

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import Tool

        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get("/tools")
            response.raise_for_status()
            return [Tool.model_validate(tool) for tool in response.json()]

    async def list_resources(self) -> Sequence["Resource"]:
        """
        List the resources available on the server.

        Retrieves the list of available resources from the /resources/read endpoint.

        Returns:
            Sequence[Resource]: A list of Resource objects available on the server

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import Resource

        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get("/resources/read")
            response.raise_for_status()
            return [Resource.model_validate(resource) for resource in response.json()]

    async def list_resource_contents(
        self, uri: str
    ) -> Sequence["TextResourceContents | BlobResourceContents"]:
        """
        List contents of a specific resource.

        Retrieves the contents of a resource identified by its URI from the
        /resources/{uri}/contents endpoint.

        Args:
            uri (str): The URI of the resource to retrieve contents for

        Returns:
            Sequence[TextResourceContents | BlobResourceContents]: A list of resource contents

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import BlobResourceContents, TextResourceContents

        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get(f"/resources/{uri}/contents")
            response.raise_for_status()
            return [
                TextResourceContents.model_validate(content)
                if content["type"] == "text"
                else BlobResourceContents.model_validate(content)
                for content in response.json()
            ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, object] | None
    ) -> "CallToolResult":
        """
        Invoke a tool on the server.

        Calls a tool with the provided arguments by making a POST request to the
        /tools/call endpoint.

        Args:
            tool_name (str): The name of the tool to call
            arguments (dict[str, object] | None): The arguments to pass to the tool

        Returns:
            CallToolResult: The result of the tool invocation

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import CallToolResult

        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            payload = {"tool_name": tool_name, "arguments": arguments or {}}
            response = await self._client.post("/tools/call", json=payload)
            response.raise_for_status()
            return CallToolResult.model_validate(response.json())
