"""
Server-Sent Events (SSE) implementation of the Model Context Protocol (MCP) server client.

This module provides a client implementation for interacting with MCP servers over
Server-Sent Events (SSE). It enables connection management, tool discovery,
resource querying, and tool execution through SSE for server-to-client streaming
and HTTP POST for client-to-server communication.

The implementation follows the MCPServerProtocol interface and uses httpx for
asynchronous HTTP communication with SSE support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol
from agentle.mcp.session_management import SessionManager, InMemorySessionManager

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
    Server-Sent Events (SSE) implementation of the MCP (Model Context Protocol) server client.

    This class provides a client implementation for interacting with remote MCP servers
    over Server-Sent Events (SSE). It uses SSE for server-to-client streaming communication
    and HTTP POST requests for client-to-server communication, following the MCP specification.

    The implementation supports session management, handles connection lifecycle,
    tool discovery, resource management, and tool execution.

    Attributes:
        server_name (str): A human-readable name for the server
        server_url (str): The base URL of the SSE server
        sse_endpoint (str): The endpoint path for SSE connections (e.g., "/sse")
        messages_endpoint (str): The endpoint path for POST messages (e.g., "/messages")
        headers (MutableMapping[str, str]): HTTP headers to include with each request
        timeout_s (float): Request timeout in seconds
        session_manager (SessionManager): Manager for storing session information

    Usage:
        server = SSEMCPServer(
            server_name="Example SSE MCP",
            server_url="http://example.com",
            sse_endpoint="/sse",
            messages_endpoint="/messages"
        )
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("tool_name", {"param": "value"})
        await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    server_url: str = Field(..., description="Base URL for the SSE MCP server")
    sse_endpoint: str | Callable[..., str] = Field(
        default="/sse",
        description="The endpoint path for SSE connections, relative to the server URL",
    )
    messages_endpoint: str | Callable[..., str] = Field(
        default="/messages",
        description="The endpoint path for POST messages, relative to the server URL",
    )

    # Optional configuration fields
    headers: MutableMapping[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to include with each request",
    )
    timeout_s: float = Field(
        default=100.0, description="Timeout in seconds for HTTP requests"
    )
    session_manager: SessionManager = Field(
        default_factory=InMemorySessionManager,
        description="Session manager for storing session state",
    )

    # Internal state
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    _client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    _sse_response: Optional[httpx.Response] = PrivateAttr(default=None)
    _sse_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)
    _pending_requests: MutableMapping[str, asyncio.Future[MutableMapping[str, Any]]] = (
        PrivateAttr(default_factory=dict)
    )
    _jsonrpc_id_counter: int = PrivateAttr(default=1)
    _initialized: bool = PrivateAttr(default=False)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _last_event_id: Optional[str] = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        """
        Get a readable name for the server.

        Returns:
            str: The human-readable server name
        """
        return self.server_name

    @property
    def _server_key(self) -> str:
        """
        Get a unique key for this server for session tracking.

        Returns:
            str: A unique identifier for this server instance
        """
        return f"{self.server_url}:{self.sse_endpoint}:{self.messages_endpoint}"

    async def _create_client(self) -> httpx.AsyncClient:
        """
        Create a new HTTP client for the current event loop.

        Returns:
            httpx.AsyncClient: A new HTTP client instance
        """
        # Set up the HTTP client with proper headers for SSE
        base_headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # Merge with user-provided headers
        all_headers = {**base_headers, **self.headers}

        self._logger.debug(f"Creating new HTTP client with headers: {all_headers}")

        # Create a fresh client bound to the current event loop
        return httpx.AsyncClient(
            base_url=str(self.server_url), timeout=self.timeout_s, headers=all_headers
        )

    async def connect_async(self) -> None:
        """
        Connect to the SSE MCP server and initialize the MCP protocol.

        Establishes an SSE connection to the server and performs the
        initialization handshake as defined in the MCP specification.

        Raises:
            ConnectionError: If the connection to the server cannot be established
        """
        self._logger.info(f"Connecting to SSE server: {self.server_url}")

        # Check if we have existing session information
        server_key = self._server_key
        session_data = await self.session_manager.get_session(server_key)

        if session_data is not None:
            self._logger.debug(f"Found existing session for {server_key}")
            self._session_id = session_data.get("session_id")
            self._last_event_id = session_data.get("last_event_id")
            self._jsonrpc_id_counter = session_data.get("jsonrpc_counter", 1)
            self._initialized = True
            return

        # Create a new client for this operation
        self._client = await self._create_client()

        try:
            # Prepare headers for SSE connection
            sse_headers: Dict[str, str] = {}
            if self._last_event_id:
                sse_headers["Last-Event-ID"] = self._last_event_id

            # Establish SSE connection
            self._logger.debug(f"Establishing SSE connection to {self.sse_endpoint}")
            self._sse_response = await self._client.get(
                self.sse_endpoint()
                if callable(self.sse_endpoint)
                else self.sse_endpoint,
                headers=sse_headers,
            )

            if self._sse_response.status_code != 200:
                raise ConnectionError(
                    f"Failed to establish SSE connection: HTTP {self._sse_response.status_code}"
                )

            # Check for session ID in response headers
            session_id = self._sse_response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
                self._logger.debug(f"Session established with ID: {session_id}")

            # Start background task to read SSE events
            self._sse_task = asyncio.create_task(self._read_sse_events())

            # Initialize the MCP protocol
            await self._initialize_protocol()

            # Store session info using session manager
            await self.session_manager.store_session(
                server_key,
                {
                    "session_id": self._session_id,
                    "last_event_id": self._last_event_id,
                    "jsonrpc_counter": self._jsonrpc_id_counter,
                },
            )

            self._initialized = True
            self._logger.info("SSE MCP protocol initialized successfully")

        except Exception as e:
            self._logger.error(f"Error connecting to SSE server: {e}")
            self._initialized = False
            await self._cleanup_connection()
            raise ConnectionError(
                f"Could not connect to SSE server {self.server_url}: {e}"
            )

    async def cleanup_async(self) -> None:
        """
        Clean up the SSE server connection.

        Closes the SSE connection and HTTP client if they exist. If a session ID was
        established, attempts to terminate the session with a DELETE request.

        Returns:
            None
        """
        self._logger.info(f"Closing connection with SSE server: {self.server_url}")

        await self._cleanup_connection()

        # If we have a session ID, try to terminate the session
        if self._session_id:
            try:
                temp_client = await self._create_client()
                headers = {"Mcp-Session-Id": self._session_id}
                await temp_client.delete(
                    self.messages_endpoint()
                    if callable(self.messages_endpoint)
                    else self.messages_endpoint,
                    headers=headers,
                )
                self._logger.debug(f"Session terminated: {self._session_id}")

                # Remove from session manager
                server_key = self._server_key
                await self.session_manager.delete_session(server_key)

                await temp_client.aclose()
            except Exception as e:
                self._logger.warning(f"Failed to terminate session: {e}")

        # Close the session manager
        await self.session_manager.close()

        self._session_id = None
        self._last_event_id = None
        self._initialized = False

    async def _cleanup_connection(self) -> None:
        """
        Clean up the current connection resources.
        """
        # Cancel the SSE reading task
        if self._sse_task is not None:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

        # Close the SSE response
        if self._sse_response is not None:
            await self._sse_response.aclose()
            self._sse_response = None

        # Close the HTTP client
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    async def _initialize_protocol(self) -> None:
        """
        Initialize the MCP protocol with the server.

        Sends the initialize request and waits for the response.
        """
        self._logger.info("Initializing MCP protocol over SSE")

        initialize_request: MutableMapping[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(self._jsonrpc_id_counter),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",  # Use an older version for better compatibility
                "clientInfo": {"name": "agentle-mcp-client", "version": "0.1.0"},
                "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
            },
        }
        self._jsonrpc_id_counter += 1

        response = await self._send_request_via_post(initialize_request)
        if "error" in response:
            raise ConnectionError(
                f"Failed to initialize MCP protocol: {response['error']}"
            )

        # Send initialized notification
        initialized_notification: MutableMapping[str, Any] = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        }
        await self._send_notification_via_post(initialized_notification)

        self._logger.info("MCP protocol initialized successfully over SSE")

    async def _read_sse_events(self) -> None:
        """
        Background task to read SSE events from the server.

        Continuously reads Server-Sent Events and processes JSON-RPC messages.
        """
        if self._sse_response is None:
            return

        try:
            async for event in self._parse_sse_stream(self._sse_response):
                data = event["data"]
                event_id = event.get("id")

                # Track the last event ID for potential resumability
                if event_id:
                    self._last_event_id = event_id

                    # Update session data
                    server_key = self._server_key
                    session_data = (
                        await self.session_manager.get_session(server_key) or {}
                    )
                    session_data["last_event_id"] = event_id
                    await self.session_manager.store_session(server_key, session_data)

                # Handle JSON-RPC message
                if isinstance(data, dict):
                    # Type cast to ensure proper typing for the message handler
                    message: MutableMapping[str, Any] = data
                    await self._handle_jsonrpc_message(message)

        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            self._logger.debug("SSE reading task cancelled")
        except Exception as e:
            self._logger.error(f"Error reading SSE events: {e}")
            # Don't re-raise to avoid crashing the connection

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

    async def _handle_jsonrpc_message(self, message: MutableMapping[str, Any]) -> None:
        """
        Handle an incoming JSON-RPC message.

        Args:
            message (MutableMapping[str, Any]): The JSON-RPC message
        """
        self._logger.debug(f"Received JSON-RPC message: {message}")

        # Handle response to a request
        if "id" in message:
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.cancelled():
                    future.set_result(message)
            else:
                self._logger.warning(
                    f"Received response for unknown request ID: {request_id}"
                )

        # Handle notification or request from server
        elif "method" in message:
            # TODO: Implement handling of server requests and notifications if needed
            self._logger.debug(f"Received notification or request: {message}")

    async def _send_request_via_post(
        self, request: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Send a JSON-RPC request via HTTP POST and wait for the response.

        Args:
            request (MutableMapping[str, Any]): The JSON-RPC request to send

        Returns:
            MutableMapping[str, Any]: The JSON-RPC response

        Raises:
            ConnectionError: If the server is not connected
            TimeoutError: If the request times out
        """
        if self._client is None:
            raise ConnectionError("Server not connected")

        # Set up a future to receive the response
        request_id = request["id"]
        response_future: asyncio.Future[MutableMapping[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Prepare headers
            headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id

            # Send the request via POST
            await self._client.post(
                self.messages_endpoint()
                if callable(self.messages_endpoint)
                else self.messages_endpoint,
                json=request,
                headers=headers,
            )

            # Wait for the response with timeout
            return await asyncio.wait_for(response_future, timeout=self.timeout_s)

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request timed out after {self.timeout_s} seconds")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise ConnectionError(f"Error sending request: {e}")

    async def _send_notification_via_post(
        self, notification: MutableMapping[str, Any]
    ) -> None:
        """
        Send a JSON-RPC notification via HTTP POST.

        Args:
            notification (MutableMapping[str, Any]): The JSON-RPC notification to send

        Raises:
            ConnectionError: If the server is not connected
        """
        if self._client is None:
            raise ConnectionError("Server not connected")

        # Prepare headers
        headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        # Send the notification via POST
        await self._client.post(
            self.messages_endpoint()
            if callable(self.messages_endpoint)
            else self.messages_endpoint,
            json=notification,
            headers=headers,
        )

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
        """
        # Ensure we're connected first
        if not self._initialized:
            self._logger.debug("Server not initialized, connecting first")
            await self.connect_async()

        if not self._initialized:
            raise ConnectionError("Failed to initialize connection")

        # Create the JSON-RPC request
        request_id = str(self._jsonrpc_id_counter)
        self._jsonrpc_id_counter += 1

        # Update the counter in session data
        server_key = self._server_key
        session_data = await self.session_manager.get_session(server_key) or {}
        session_data["jsonrpc_counter"] = self._jsonrpc_id_counter
        await self.session_manager.store_session(server_key, session_data)

        request: MutableMapping[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        try:
            response = await self._send_request_via_post(request)

            if "error" in response:
                raise ValueError(f"JSON-RPC error: {response['error']}")

            return response

        except ConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            # Handle other exceptions
            self._logger.error(f"Error sending request: {e}")
            raise ConnectionError(f"Error sending request: {e}")

    async def list_tools_async(self) -> Sequence[Tool]:
        """
        List the tools available on the server.

        Returns:
            Sequence[Tool]: A list of Tool objects available on the server

        Raises:
            ConnectionError: If the server is not connected
        """
        from mcp.types import Tool

        response = await self._send_request("tools/list")

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
        """
        from mcp.types import Resource

        response = await self._send_request("resources/list")

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
        """
        from mcp.types import BlobResourceContents, TextResourceContents

        response = await self._send_request("resources/read", {"uri": uri})

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
        """
        from mcp.types import CallToolResult

        response = await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments or {}}
        )

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        return CallToolResult.model_validate(response["result"])
