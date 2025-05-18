"""
HTTP SSE implementation of the Model Context Protocol (MCP) server client.

This module provides an HTTP client implementation for interacting with MCP servers
using Server-Sent Events (SSE) for streaming responses where appropriate.
It enables connection management, tool discovery, resource querying, and tool execution
through standard HTTP endpoints.

The implementation follows the MCPServerProtocol interface and uses httpx for
asynchronous HTTP communication.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, override

import httpx
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
    HTTP SSE implementation of the MCP (Model Context Protocol) server client.

    This class provides a client implementation for interacting with remote MCP servers
    over HTTP with Server-Sent Events (SSE) for streaming responses. It handles
    connection management, tool discovery, resource management, and tool invocation
    through HTTP endpoints.

    Attributes:
        server_name (str): A human-readable name for the server
        server_url (AnyUrl): The base URL of the HTTP server
        headers (dict[str, str]): HTTP headers to include with each request
        timeout_s (float): Request timeout in seconds

    Usage:
        server = SSEMCPServer(server_name="Example MCP", server_url="http://example.com/api")
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("tool_name", {"param": "value"})
        await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    server_url: str = Field(..., description="Base URL for the HTTP MCP server")

    # Optional configuration fields
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to include with each request",
    )
    timeout_s: float = Field(
        default=100.0, description="Timeout in seconds for HTTP requests"
    )

    # Internal state
    _client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )

    @override
    async def connect_async(self) -> None:
        """
        Connect to the HTTP MCP server.

        Establishes an HTTP client connection to the server and verifies connectivity
        by performing a test request to the root endpoint.

        Raises:
            ConnectionError: If the connection to the server cannot be established
        """
        self._logger.info(f"Connecting to HTTP server: {self.server_url}")

        # Set up the HTTP client with proper headers for SSE
        sse_headers = self.headers.copy()
        sse_headers.update(
            {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            }
        )

        self._client = httpx.AsyncClient(
            base_url=str(self.server_url), timeout=self.timeout_s, headers=sse_headers
        )

        # Verify connection with the server
        try:
            response = await self._client.get("/")
            if response.status_code != 200:
                self._logger.warning(
                    f"Server responded with status {response.status_code}"
                )
        except Exception as e:
            self._logger.error(f"Error connecting to server: {e}")
            await self.cleanup_async()
            raise ConnectionError(f"Could not connect to server {self.server_url}: {e}")

    @property
    @override
    def name(self) -> str:
        """
        Get a readable name for the server.

        Returns:
            str: The human-readable server name
        """
        return self.server_name

    @override
    async def cleanup_async(self) -> None:
        """
        Cleanup the server connection.

        Closes the HTTP client connection if it exists. This method should be called
        when the server connection is no longer needed.
        """
        if self._client is not None:
            self._logger.info(f"Closing connection with HTTP server: {self.server_url}")
            await self._client.aclose()
            self._client = None

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Parse an SSE stream from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response with the SSE stream

        Yields:
            Dict[str, Any]: Parsed SSE events as dictionaries
        """
        event_data = ""
        event_id = None
        event_type = None

        async for line in response.aiter_lines():
            line = line.rstrip("\n")
            if not line:
                # End of event, yield if we have data
                if event_data:
                    try:
                        data = json.loads(event_data)
                        yield {
                            "id": event_id,
                            "type": event_type or "message",
                            "data": data,
                        }
                    except json.JSONDecodeError:
                        yield {
                            "id": event_id,
                            "type": event_type or "message",
                            "data": event_data,
                        }

                    # Reset for next event
                    event_data = ""
                    event_id = None
                    event_type = None
                continue

            if line.startswith(":"):
                # Comment, ignore
                continue

            # Parse field:value format
            match = re.match(r"([^:]+)(?::(.*))?", line)
            if match:
                field, value = match.groups()
                value = value.lstrip() if value else ""

                if field == "data":
                    event_data += value + "\n"
                elif field == "id":
                    event_id = value
                elif field == "event":
                    event_type = value

    async def list_tools_async(self) -> Sequence[Tool]:
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
            raise ConnectionError("Server not connected")

        try:
            response = await self._client.get("/tools")
            response.raise_for_status()
            return [Tool.model_validate(tool) for tool in response.json()]
        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise

    async def list_resources_async(self) -> Sequence[Resource]:
        """
        List the resources available on the server.

        Retrieves the list of available resources from the /resources endpoint.

        Returns:
            Sequence[Resource]: A list of Resource objects available on the server

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import Resource

        if self._client is None:
            raise ConnectionError("Server not connected")

        try:
            response = await self._client.get("/resources")
            response.raise_for_status()
            return [Resource.model_validate(resource) for resource in response.json()]
        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise

    async def list_resource_contents_async(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
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
            raise ConnectionError("Server not connected")

        try:
            response = await self._client.get(f"/resources/{uri}/contents")
            response.raise_for_status()
            return [
                TextResourceContents.model_validate(content)
                if content["type"] == "text"
                else BlobResourceContents.model_validate(content)
                for content in response.json()
            ]
        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise

    async def call_tool_async(
        self, tool_name: str, arguments: dict[str, object] | None
    ) -> "CallToolResult":
        """
        Invoke a tool on the server.

        Calls a tool with the provided arguments by making a POST request to the
        /tools/call endpoint. Supports both regular responses and SSE streaming responses.

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
            raise ConnectionError("Server not connected")

        try:
            payload = {"tool_name": tool_name, "arguments": arguments or {}}

            # Check if tool requires streaming response
            tools = await self.list_tools_async()
            tool = next((t for t in tools if t.name == tool_name), None)

            if tool and getattr(tool, "streaming", False):
                # Use SSE for streaming tools
                response = await self._client.post(
                    "/tools/call/stream",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                )
                response.raise_for_status()

                # Collect all events and consolidate the result
                final_result = None
                async for event in self._parse_sse_stream(response):
                    if event["type"] == "result":
                        final_result = event["data"]
                    elif event["type"] == "error":
                        raise ValueError(f"Tool execution error: {event['data']}")

                if final_result:
                    return CallToolResult.model_validate(final_result)
                raise ValueError("No result received from streaming tool execution")
            else:
                # Use regular HTTP for non-streaming tools
                response = await self._client.post("/tools/call", json=payload)
                response.raise_for_status()
                return CallToolResult.model_validate(response.json())
        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise
