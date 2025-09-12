"""
Server-Sent Events (SSE) implementation of the Model Context Protocol (MCP) server client.

FIXED VERSION for aiohttp with proper SSE protocol implementation:
- Correct endpoint discovery mechanism
- Proper sessionId handling in POST requests (FIXED: No duplicate sessionId parameters)
- Fixed SSE parsing for aiohttp
- Connection state management fixes
- Proper protocol implementation matching the Node.js server
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import aiohttp
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

    FIXED VERSION with proper SSE protocol implementation for aiohttp.

    Key fixes:
    - Proper endpoint discovery mechanism
    - SessionId extraction and usage in POST requests (NO DUPLICATE sessionId parameters)
    - Correct SSE parsing with aiohttp
    - Connection state management
    - Protocol compliance with MCP SSE specification
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    server_url: str = Field(..., description="Base URL for the SSE MCP server")
    sse_endpoint: str | Callable[..., str] = Field(
        default="/sse",
        description="The endpoint path for SSE connections, relative to the server URL",
    )
    messages_endpoint: str | Callable[..., str] = Field(
        default="/message",  # Changed to match your server
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
    max_reconnect_attempts: int = Field(
        default=3, description="Maximum number of reconnection attempts"
    )
    reconnect_delay_s: float = Field(
        default=1.0, description="Delay between reconnection attempts"
    )

    # Internal state
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    _sse_response: Optional[aiohttp.ClientResponse] = PrivateAttr(default=None)
    _sse_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)
    _pending_requests: MutableMapping[str, asyncio.Future[MutableMapping[str, Any]]] = (
        PrivateAttr(default_factory=dict)
    )
    _jsonrpc_id_counter: int = PrivateAttr(default=1)
    _initialized: bool = PrivateAttr(default=False)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _last_event_id: Optional[str] = PrivateAttr(default=None)
    _dynamic_messages_endpoint: Optional[str] = PrivateAttr(default=None)
    _connection_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _is_reconnecting: bool = PrivateAttr(default=False)
    _endpoint_discovered: bool = PrivateAttr(default=False)

    @property
    def name(self) -> str:
        """Get a readable name for the server."""
        return self.server_name

    @property
    def _server_key(self) -> str:
        """Get a unique key for this server for session tracking."""
        return f"{self.server_url}:{self.sse_endpoint}:{self.messages_endpoint}"

    def _is_connected(self) -> bool:
        """Check if the server is properly connected."""
        return (
            self._client is not None
            and not self._client.closed
            and self._sse_response is not None
            and not self._sse_response.closed
            and self._session_id is not None
        )

    async def _create_client(self) -> aiohttp.ClientSession:
        """Create a new HTTP client for the current event loop."""
        base_headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        all_headers = {**base_headers, **self.headers}
        self._logger.debug(f"Creating new HTTP client with headers: {all_headers}")

        return aiohttp.ClientSession(
            headers=all_headers, timeout=aiohttp.ClientTimeout(total=self.timeout_s)
        )

    async def connect_async(self) -> None:
        """
        Connect to the SSE MCP server and initialize the MCP protocol.

        FIXED: Now properly establishes connection even when session data exists.
        """
        async with self._connection_lock:
            if self._is_connected():
                return

            self._logger.info(f"Connecting to SSE server: {self.server_url}")

            # Load session data but don't skip connection establishment
            server_key = self._server_key
            session_data = await self.session_manager.get_session(server_key)

            if session_data is not None:
                self._logger.debug(f"Found existing session for {server_key}")
                self._session_id = session_data.get("session_id")
                self._last_event_id = session_data.get("last_event_id")
                self._jsonrpc_id_counter = session_data.get("jsonrpc_counter", 1)
                self._dynamic_messages_endpoint = session_data.get("messages_endpoint")

            # Always establish a real connection
            await self._establish_connection()

    async def _establish_connection(self) -> None:
        """Establish the actual SSE connection and initialize protocol."""
        try:
            # Clean up any existing connection
            await self._cleanup_connection()

            # Create new client
            self._client = await self._create_client()

            # Prepare headers for SSE connection
            sse_headers: Dict[str, str] = {}
            if self._last_event_id:
                sse_headers["Last-Event-ID"] = self._last_event_id

            # Establish SSE connection
            sse_url = (
                self.sse_endpoint()
                if callable(self.sse_endpoint)
                else self.sse_endpoint
            )

            full_url = urljoin(self.server_url, sse_url.lstrip("/"))
            self._logger.debug(f"Establishing SSE connection to {full_url}")

            self._sse_response = await self._client.get(full_url, headers=sse_headers)

            if self._sse_response.status != 200:
                raise ConnectionError(
                    f"Failed to establish SSE connection: HTTP {self._sse_response.status}"
                )

            # Check for session ID in response headers
            session_id = self._sse_response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
                self._logger.debug(f"Session established with ID: {session_id}")

            # Start background task to read SSE events
            self._sse_task = asyncio.create_task(self._read_sse_events())

            # Wait for endpoint discovery
            await self._wait_for_endpoint_discovery()

            # Initialize the MCP protocol if not already done
            if not self._initialized:
                await self._initialize_protocol()

            # Store session info
            await self._store_session_data()

            self._initialized = True
            self._logger.info("SSE MCP protocol initialized successfully")

        except Exception as e:
            self._logger.error(f"Error establishing connection: {e}")
            self._initialized = False
            await self._cleanup_connection()
            raise ConnectionError(
                f"Could not connect to SSE server {self.server_url}: {e}"
            )

    async def _wait_for_endpoint_discovery(self, timeout: float = 10.0) -> None:
        """Wait for the server to send endpoint discovery event."""
        start_time = asyncio.get_event_loop().time()

        while not self._endpoint_discovered:
            if asyncio.get_event_loop().time() - start_time > timeout:
                self._logger.warning(
                    "Endpoint discovery timeout, using default endpoint"
                )
                break
            await asyncio.sleep(0.1)

    async def cleanup_async(self) -> None:
        """Clean up the SSE server connection."""
        async with self._connection_lock:
            self._logger.info(f"Closing connection with SSE server: {self.server_url}")

            await self._cleanup_connection()

            # Terminate session if we have a session ID
            if self._session_id:
                await self._terminate_session()

            # Close the session manager
            await self.session_manager.close()

            self._session_id = None
            self._last_event_id = None
            self._dynamic_messages_endpoint = None
            self._initialized = False
            self._endpoint_discovered = False

    async def _cleanup_connection(self) -> None:
        """Clean up the current connection resources."""
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
            self._sse_response.close()
            self._sse_response = None

        # Close the HTTP client
        if self._client is not None:
            await self._client.close()
            self._client = None

        # Cancel any pending requests with proper cleanup
        for _, future in list(self._pending_requests.items()):
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    async def _terminate_session(self) -> None:
        """Attempt to terminate the session with the server."""
        if not self._session_id:
            return

        try:
            temp_client = await self._create_client()

            # Build the DELETE URL - use the dynamic endpoint if available
            if self._dynamic_messages_endpoint:
                # If dynamic endpoint already has sessionId, use it as-is
                if "sessionId=" in self._dynamic_messages_endpoint:
                    delete_url = urljoin(
                        self.server_url, self._dynamic_messages_endpoint.lstrip("/")
                    )
                else:
                    delete_url = urljoin(
                        self.server_url, self._dynamic_messages_endpoint.lstrip("/")
                    )
                    delete_url += f"?sessionId={self._session_id}"
            else:
                # Use default endpoint
                endpoint = (
                    self.messages_endpoint()
                    if callable(self.messages_endpoint)
                    else self.messages_endpoint
                )
                delete_url = urljoin(self.server_url, endpoint.lstrip("/"))
                delete_url += f"?sessionId={self._session_id}"

            async with temp_client.delete(delete_url) as response:
                self._logger.debug(
                    f"Session terminated: {self._session_id} with status {response.status}"
                )

            # Remove from session manager
            server_key = self._server_key
            await self.session_manager.delete_session(server_key)

            await temp_client.close()
        except Exception as e:
            self._logger.warning(f"Failed to terminate session: {e}")

    async def _store_session_data(self) -> None:
        """Store current session data."""
        server_key = self._server_key
        session_data = {
            "session_id": self._session_id,
            "last_event_id": self._last_event_id,
            "jsonrpc_counter": self._jsonrpc_id_counter,
            "messages_endpoint": self._dynamic_messages_endpoint,
        }
        await self.session_manager.store_session(server_key, session_data)

    async def _initialize_protocol(self) -> None:
        """Initialize the MCP protocol with the server."""
        self._logger.info("Initializing MCP protocol over SSE")

        initialize_request: MutableMapping[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(self._jsonrpc_id_counter),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
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

        FIXED: Proper aiohttp SSE parsing and endpoint discovery.
        """
        if self._sse_response is None:
            return

        try:
            async for event in self._parse_sse_stream(self._sse_response):
                await self._handle_sse_event(event)

        except asyncio.CancelledError:
            self._logger.debug("SSE reading task cancelled")
        except Exception as e:
            self._logger.error(f"Error reading SSE events: {e}")

            # Attempt reconnection if not already reconnecting
            if not self._is_reconnecting:
                asyncio.create_task(self._attempt_reconnection())

    async def _handle_sse_event(self, event: MutableMapping[str, Any]) -> None:
        """Handle individual SSE events."""
        data = event["data"]
        event_id = event.get("id")
        event_type = event.get("type", "message")

        # Track the last event ID for potential resumability
        if event_id:
            self._last_event_id = event_id
            await self._store_session_data()

        # Handle endpoint discovery
        if event_type == "endpoint":
            if isinstance(data, str):
                # Store the endpoint exactly as received from server
                self._dynamic_messages_endpoint = data

                # Parse the URL to extract sessionId if it's in query params
                parsed = urlparse(data)
                if parsed.query:
                    query_params = parse_qs(parsed.query)
                    if "sessionId" in query_params:
                        self._session_id = query_params["sessionId"][0]
                        self._logger.debug(
                            f"Extracted sessionId from endpoint: {self._session_id}"
                        )

                self._endpoint_discovered = True
                self._logger.debug(f"Discovered messages endpoint: {data}")
            return

        # Handle JSON-RPC message
        if isinstance(data, dict):
            message: dict[str, Any] = data
            await self._handle_jsonrpc_message(message)

    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._is_reconnecting:
            return

        self._is_reconnecting = True

        try:
            for attempt in range(self.max_reconnect_attempts):
                self._logger.info(
                    f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}"
                )

                try:
                    await asyncio.sleep(self.reconnect_delay_s * (2**attempt))
                    await self._establish_connection()
                    self._logger.info("Reconnection successful")
                    return
                except Exception as e:
                    self._logger.warning(
                        f"Reconnection attempt {attempt + 1} failed: {e}"
                    )

            self._logger.error("All reconnection attempts failed")
            self._initialized = False

        finally:
            self._is_reconnecting = False

    async def _parse_sse_stream(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[MutableMapping[str, Any]]:
        """
        Parse an SSE stream from an HTTP response.

        FIXED: Proper aiohttp line-by-line reading for SSE streams.
        """
        event_data = ""
        event_id = None
        event_type = None

        # FIXED: Use proper aiohttp line-by-line reading for SSE
        while True:
            try:
                line_bytes = await response.content.readline()
                if not line_bytes:
                    # End of stream
                    break

                line = line_bytes.decode("utf-8").rstrip("\n\r")
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._logger.error(f"Error reading SSE line: {e}")
                break

            if not line:
                # End of event, yield if we have data
                if event_data:
                    # Remove trailing newline if present
                    event_data = event_data.rstrip("\n")

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
                    # FIXED: Proper newline handling
                    if event_data:
                        event_data += "\n" + value
                    else:
                        event_data = value
                elif field == "id":
                    event_id = value
                elif field == "event":
                    event_type = value

    async def _handle_jsonrpc_message(self, message: MutableMapping[str, Any]) -> None:
        """Handle an incoming JSON-RPC message."""
        self._logger.debug(f"Received JSON-RPC message: {message}")

        # Handle response to a request
        if "id" in message:
            request_id = message["id"]
            future = self._pending_requests.pop(request_id, None)
            if future and not future.cancelled():
                future.set_result(message)
            elif not future:
                self._logger.warning(
                    f"Received response for unknown request ID: {request_id}"
                )

        # Handle notification or request from server
        elif "method" in message:
            self._logger.debug(f"Received notification or request: {message}")

    def _build_post_url(self, base_endpoint: str) -> str:
        """
        Build the POST URL correctly, avoiding duplicate sessionId parameters.

        FIXED: This is the key fix - no more duplicate sessionId parameters!
        """
        # If we have a dynamic endpoint from the server, use it as-is
        if self._dynamic_messages_endpoint:
            return urljoin(self.server_url, self._dynamic_messages_endpoint.lstrip("/"))

        # Otherwise, build URL with the base endpoint
        post_url = urljoin(self.server_url, base_endpoint.lstrip("/"))

        # Only add sessionId if it's not already in the URL and we have a session ID
        if self._session_id and "sessionId=" not in post_url:
            separator = "&" if "?" in post_url else "?"
            post_url += f"{separator}sessionId={self._session_id}"

        return post_url

    async def _send_request_via_post(
        self, request: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Send a JSON-RPC request via HTTP POST and wait for the response.

        FIXED: Proper sessionId handling and connection validation - no duplicate parameters!
        """
        if not self._is_connected() or self._client is None:
            raise ConnectionError("Server not connected")

        # Set up a future to receive the response
        request_id = str(request["id"])
        response_future: asyncio.Future[MutableMapping[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Prepare headers
            headers: Dict[str, str] = {"Content-Type": "application/json"}

            # Build the correct POST URL (FIXED: No duplicate sessionId)
            base_endpoint = (
                self.messages_endpoint()
                if callable(self.messages_endpoint)
                else self.messages_endpoint
            )
            post_url = self._build_post_url(base_endpoint)

            self._logger.debug(f"Sending POST request to: {post_url}")

            # Send the request via POST
            async with self._client.post(
                post_url, json=request, headers=headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise ConnectionError(
                        f"HTTP error: {response.status} - {error_text}"
                    )

            # Wait for the response with timeout
            return await asyncio.wait_for(response_future, timeout=self.timeout_s)

        except asyncio.TimeoutError:
            # FIXED: Proper cleanup of timed out requests
            future = self._pending_requests.pop(request_id, None)
            if future and not future.cancelled():
                future.cancel()
            raise TimeoutError(f"Request timed out after {self.timeout_s} seconds")
        except Exception as e:
            # Clean up on any error
            future = self._pending_requests.pop(request_id, None)
            if future and not future.cancelled():
                future.cancel()
            raise ConnectionError(f"Error sending request: {e}")

    async def _send_notification_via_post(
        self, notification: MutableMapping[str, Any]
    ) -> None:
        """Send a JSON-RPC notification via HTTP POST."""
        if not self._is_connected() or self._client is None:
            raise ConnectionError("Server not connected")

        # Prepare headers
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        # Build the correct POST URL (FIXED: No duplicate sessionId)
        base_endpoint = (
            self.messages_endpoint()
            if callable(self.messages_endpoint)
            else self.messages_endpoint
        )
        post_url = self._build_post_url(base_endpoint)

        # Send the notification via POST
        async with self._client.post(
            post_url, json=notification, headers=headers
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise ConnectionError(f"HTTP error: {response.status} - {error_text}")

    async def _send_request(
        self, method: str, params: Optional[MutableMapping[str, Any]] = None
    ) -> MutableMapping[str, Any]:
        """Send a JSON-RPC request to the server with automatic reconnection."""
        # Ensure we're connected first
        if not self._is_connected():
            self._logger.debug("Server not connected, connecting first")
            await self.connect_async()

        if not self._is_connected():
            raise ConnectionError("Failed to establish connection")

        # Create the JSON-RPC request
        request_id = str(self._jsonrpc_id_counter)
        self._jsonrpc_id_counter += 1

        # Update the counter in session data
        await self._store_session_data()

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

    # Protocol methods (unchanged from previous version)
    async def list_tools_async(self) -> Sequence[Tool]:
        """List the tools available on the server."""
        from mcp.types import Tool

        response = await self._send_request("tools/list")

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        if "tools" not in response["result"]:
            raise ValueError("Invalid response format: missing 'tools' in result")

        return [Tool.model_validate(tool) for tool in response["result"]["tools"]]

    async def list_resources_async(self) -> Sequence[Resource]:
        """List the resources available on the server."""
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
        """List contents of a specific resource."""
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
        """Invoke a tool on the server."""
        from mcp.types import CallToolResult

        response = await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments or {}}
        )

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        return CallToolResult.model_validate(response["result"])
