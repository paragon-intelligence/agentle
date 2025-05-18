"""
Streamable HTTP implementation of the Model Context Protocol (MCP) server client.

This module provides an HTTP client implementation for interacting with MCP servers
using the Streamable HTTP transport as defined in the MCP 2025-03-26 specification.
It enables connection management, tool discovery, resource querying, and tool execution
through a standardized MCP endpoint.

The implementation follows the MCPServerProtocol interface and uses httpx for
asynchronous HTTP communication.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

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


class StreamableHTTPMCPServer(MCPServerProtocol):
    """
    Streamable HTTP implementation of the MCP (Model Context Protocol) server client.

    This class provides a client implementation for interacting with remote MCP servers
    over HTTP using the Streamable HTTP transport (MCP 2025-03-26 spec). It supports
    both regular and streaming responses, session management, and handles connection
    management, tool discovery, resource management, and tool execution.

    Attributes:
        server_name (str): A human-readable name for the server
        server_url (AnyUrl): The base URL of the HTTP server
        mcp_endpoint (str): The endpoint path for MCP requests (e.g., "/mcp")
        headers (MutableMapping[str, str]): HTTP headers to include with each request
        timeout_s (float): Request timeout in seconds

    Usage:
        server = StreamableHTTPMCPServer(server_name="Example MCP", server_url="http://example.com", mcp_endpoint="/mcp")
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("tool_name", {"param": "value"})
        await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    server_url: str = Field(..., description="Base URL for the HTTP MCP server")
    mcp_endpoint: str = Field(
        default="/mcp",
        description="The endpoint path for MCP requests, relative to the server URL",
    )

    # Optional configuration fields
    headers: MutableMapping[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to include with each request",
    )
    timeout_s: float = Field(
        default=100.0, description="Timeout in seconds for HTTP requests"
    )

    # Internal state
    _client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    _session_id: Optional[str] = PrivateAttr(default=None)
    _last_event_id: Optional[str] = PrivateAttr(default=None)
    _jsonrpc_id_counter: int = PrivateAttr(default=1)

    @property
    def name(self) -> str:
        """
        Get a readable name for the server.

        Returns:
            str: The human-readable server name
        """
        return self.server_name

    async def connect_async(self) -> None:
        """
        Connect to the HTTP MCP server and initialize the MCP protocol.

        Establishes an HTTP client connection to the server and performs the
        initialization handshake as defined in the MCP specification.

        Raises:
            ConnectionError: If the connection to the server cannot be established
        """
        self._logger.info(f"Connecting to HTTP server: {self.server_url}")

        # Set up the HTTP client with proper headers for Streamable HTTP
        base_headers = {
            "Accept": "application/json, text/event-stream",
            "Cache-Control": "no-cache",
        }

        # Merge with user-provided headers
        all_headers = {**base_headers, **self.headers}

        self._client = httpx.AsyncClient(
            base_url=str(self.server_url), timeout=self.timeout_s, headers=all_headers
        )

        # Initialize the MCP protocol
        try:
            # Send initialization request
            initialize_request: MutableMapping[str, Any] = {
                "jsonrpc": "2.0",
                "id": str(self._jsonrpc_id_counter),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "clientInfo": {"name": "agentle-mcp-client", "version": "0.1.0"},
                    "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
                },
            }
            self._jsonrpc_id_counter += 1

            # POST the initialize request to the MCP endpoint
            response = await self._client.post(
                self.mcp_endpoint, json=initialize_request
            )

            if response.status_code != 200:
                self._logger.warning(
                    f"Server responded with status {response.status_code}"
                )
                raise ConnectionError(
                    f"Failed to initialize: HTTP {response.status_code}"
                )

            # Parse the response
            init_result = response.json()
            if "error" in init_result:
                raise ConnectionError(f"Failed to initialize: {init_result['error']}")

            # Check for session ID in headers
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
                self._logger.debug(f"Session established with ID: {session_id}")

            # Send initialized notification
            await self._send_notification(
                {"jsonrpc": "2.0", "method": "initialized", "params": {}}
            )

            self._logger.info("MCP protocol initialized successfully")

        except Exception as e:
            self._logger.error(f"Error connecting to server: {e}")
            await self.cleanup_async()
            raise ConnectionError(f"Could not connect to server {self.server_url}: {e}")

    async def cleanup_async(self) -> None:
        """
        Clean up the server connection.

        Closes the HTTP client connection if it exists. If a session ID was
        established, attempts to terminate the session with a DELETE request.

        Returns:
            None
        """
        if self._client is not None:
            self._logger.info(f"Closing connection with HTTP server: {self.server_url}")

            # If we have a session ID, try to terminate the session
            if self._session_id:
                try:
                    headers = {"Mcp-Session-Id": self._session_id}
                    await self._client.delete(self.mcp_endpoint, headers=headers)
                    self._logger.debug(f"Session terminated: {self._session_id}")
                except Exception as e:
                    self._logger.warning(f"Failed to terminate session: {e}")

            await self._client.aclose()
            self._client = None
            self._session_id = None
            self._last_event_id = None

    async def list_tools_async(self) -> Sequence[Tool]:
        """
        List the tools available on the server.

        Returns:
            Sequence[Tool]: A list of Tool objects available on the server

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import Tool

        response = await self._send_request("listTools")

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        if "tools" not in response["result"]:
            raise ValueError("Invalid response format: missing 'tools' in result")

        return [Tool.model_validate(tool) for tool in response["result"]["tools"]]

    async def list_resources_async(self) -> Sequence[Resource]:
        """
        List the resources available on the server.

        Returns:
            Sequence[Resource]: A list of Resource objects available on the server

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import Resource

        response = await self._send_request("listResources")

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        if "resources" not in response["result"]:
            raise ValueError("Invalid response format: missing 'resources' in result")

        return [
            Resource.model_validate(resource)
            for resource in response["result"]["resources"]
        ]

    async def list_resource_contents_async(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
        """
        List contents of a specific resource.

        Args:
            uri (str): The URI of the resource to retrieve contents for

        Returns:
            Sequence[TextResourceContents | BlobResourceContents]: A list of resource contents

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import BlobResourceContents, TextResourceContents

        response = await self._send_request("readResource", {"uri": uri})

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        if "contents" not in response["result"]:
            raise ValueError("Invalid response format: missing 'contents' in result")

        return [
            TextResourceContents.model_validate(content)
            if content["type"] == "text"
            else BlobResourceContents.model_validate(content)
            for content in response["result"]["contents"]
        ]

    async def call_tool_async(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> CallToolResult:
        """
        Invoke a tool on the server.

        Args:
            tool_name (str): The name of the tool to call
            arguments (MutableMapping[str, object] | None): The arguments to pass to the tool

        Returns:
            CallToolResult: The result of the tool invocation

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import CallToolResult

        response = await self._send_request(
            "callTool", {"tool": tool_name, "arguments": arguments or {}}
        )

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        return CallToolResult.model_validate(response["result"])

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[MutableMapping[str, Any]]:
        """
        Parse an SSE stream from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response with the SSE stream

        Yields:
            MutableMapping[str, Any]: Parsed SSE events as dictionaries
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

                        # Track the last event ID for potential resumability
                        if event_id:
                            self._last_event_id = event_id
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

    async def _send_request(
        self, method: str, params: Optional[MutableMapping[str, Any]] = None
    ) -> MutableMapping[str, Any]:
        """
        Send a JSON-RPC request to the server.

        Args:
            method (str): The JSON-RPC method to call
            params (MutableMapping[str, Any], optional): The parameters for the method

        Returns:
            MutableMapping[str, Any]: The JSON-RPC response

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        if self._client is None:
            raise ConnectionError("Server not connected")

        # Create the JSON-RPC request
        request_id = str(self._jsonrpc_id_counter)
        self._jsonrpc_id_counter += 1

        request: MutableMapping[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        # Prepare headers
        headers: MutableMapping[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        # Send the request
        try:
            response = await self._client.post(
                self.mcp_endpoint, json=request, headers=headers
            )

            # Check for session ID in response
            if "Mcp-Session-Id" in response.headers and not self._session_id:
                self._session_id = response.headers["Mcp-Session-Id"]
                self._logger.debug(f"Session established with ID: {self._session_id}")

            # Handle different response types
            if response.status_code == 404 and self._session_id:
                # Session expired, we need to reconnect
                self._logger.warning("Session expired, reconnecting...")
                self._session_id = None
                await self.connect_async()  # Reconnect
                return await self._send_request(method, params)  # Retry the request

            elif response.status_code != 200:
                raise ConnectionError(
                    f"Server returned error: HTTP {response.status_code}"
                )

            content_type = response.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                # This is an SSE stream
                self._logger.debug("Received SSE stream response")

                # Process the stream and find the response for our request
                async for event in self._parse_sse_stream(response):
                    data = event["data"]

                    # Check if this is the response to our request
                    if (
                        isinstance(data, dict)
                        and "id" in data
                        and data["id"] == request_id
                    ):
                        if "error" in data:
                            raise ValueError(f"JSON-RPC error: {data['error']}")
                        # Create a new dictionary with explicit typing
                        result: MutableMapping[str, Any] = {}
                        for k, v in data.items():
                            result[k] = v
                        return result

                raise ValueError("Did not receive response for request in SSE stream")

            elif "application/json" in content_type:
                # This is a direct JSON response
                data_raw = response.json()

                # Create a new dictionary with explicit typing
                _data: MutableMapping[str, Any] = {}
                for k, v in data_raw.items():
                    _data[k] = v

                if "error" in _data:
                    raise ValueError(f"JSON-RPC error: {_data['error']}")

                return _data

            else:
                raise ValueError(f"Unexpected content type: {content_type}")

        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise

    async def _send_notification(self, notification: MutableMapping[str, Any]) -> None:
        """
        Send a JSON-RPC notification to the server.

        Args:
            notification (MutableMapping[str, Any]): The JSON-RPC notification to send

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        if self._client is None:
            raise ConnectionError("Server not connected")

        # Prepare headers
        headers: MutableMapping[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        # Send the notification
        try:
            response = await self._client.post(
                self.mcp_endpoint, json=notification, headers=headers
            )

            # Expect 202 Accepted for notifications
            if response.status_code != 202:
                self._logger.warning(
                    f"Unexpected status code for notification: {response.status_code}"
                )

            # Check for session ID in response
            if "Mcp-Session-Id" in response.headers and not self._session_id:
                self._session_id = response.headers["Mcp-Session-Id"]
                self._logger.debug(f"Session established with ID: {self._session_id}")

        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise

    async def _open_sse_stream(self) -> httpx.Response:
        """
        Open an SSE stream for server-initiated messages.

        Returns:
            httpx.Response: The HTTP response with the SSE stream

        Raises:
            ConnectionError: If the server is not connected
            httpx.RequestError: If there's an error during the HTTP request
        """
        if self._client is None:
            raise ConnectionError("Server not connected")

        # Prepare headers
        headers: MutableMapping[str, str] = {"Accept": "text/event-stream"}

        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        if self._last_event_id:
            headers["Last-Event-ID"] = self._last_event_id

        # Open the SSE stream
        try:
            response = await self._client.get(self.mcp_endpoint, headers=headers)

            if response.status_code == 405:
                self._logger.warning(
                    "Server does not support SSE streams via GET method"
                )
                raise ValueError("Server does not support SSE streams via GET method")

            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to open SSE stream: HTTP {response.status_code}"
                )

            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise ValueError(
                    f"Expected text/event-stream response, got {content_type}"
                )

            return response

        except httpx.RequestError as e:
            self._logger.error(f"HTTP request error: {e}")
            raise
