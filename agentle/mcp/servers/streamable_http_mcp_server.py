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
        session_manager (SessionManager): Manager for storing session information

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
    mcp_endpoint: str | Callable[..., str] = Field(
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
    session_manager: SessionManager = Field(
        default_factory=InMemorySessionManager,
        description="Session manager for storing session state",
    )

    # Internal state
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    # Session state stored as primitives
    _session_id: Optional[str] = PrivateAttr(default=None)
    _last_event_id: Optional[str] = PrivateAttr(default=None)
    _jsonrpc_id_counter: int = PrivateAttr(default=1)
    _initialized: bool = PrivateAttr(default=False)

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
        return f"{self.server_url}:{self.mcp_endpoint}"

    async def _create_client(self) -> httpx.AsyncClient:
        """
        Create a new HTTP client for the current event loop.

        Returns:
            httpx.AsyncClient: A new HTTP client instance
        """
        # Set up the HTTP client with proper headers for Streamable HTTP
        base_headers = {
            "Accept": "application/json, text/event-stream",
            "Cache-Control": "no-cache",
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
        Connect to the HTTP MCP server and initialize the MCP protocol.

        Establishes an HTTP client connection to the server and performs the
        initialization handshake as defined in the MCP specification.

        Raises:
            ConnectionError: If the connection to the server cannot be established
        """
        self._logger.info(f"Connecting to HTTP server: {self.server_url}")

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
        client = await self._create_client()

        # Initialize the MCP protocol
        try:
            # Send initialization request
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

            # POST the initialize request to the MCP endpoint
            response = await client.post(
                self.mcp_endpoint()
                if callable(self.mcp_endpoint)
                else self.mcp_endpoint,
                json=initialize_request,
            )

            self._logger.debug(
                f"Initialization response status: {response.status_code}"
            )
            self._logger.debug(f"Initialization response headers: {response.headers}")
            self._logger.debug(
                f"Initialization response content: {response.text[:100]}..."
            )

            if response.status_code != 200:
                self._logger.warning(
                    f"Server responded with status {response.status_code}"
                )
                raise ConnectionError(
                    f"Failed to initialize: HTTP {response.status_code}"
                )

            # Check the content type before parsing
            content_type = response.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                # Handle SSE stream response
                self._logger.debug("Received SSE stream response during initialization")
                # Process the stream to find the initialization response
                async for event in self._parse_sse_stream(response):
                    data = event["data"]
                    # Check if this is the response to our initialization request
                    if (
                        isinstance(data, dict)
                        and "id" in data
                        and data["id"] == str(self._jsonrpc_id_counter - 1)
                    ):
                        if "error" in data:
                            raise ConnectionError(
                                f"Failed to initialize: {data['error']}"
                            )
                        init_result = data
                        break
                else:
                    # If we didn't find a matching response
                    raise ConnectionError(
                        "Did not receive initialization response in SSE stream"
                    )
            elif "application/json" in content_type:
                # Parse the JSON response
                try:
                    init_result = response.json()
                    if "error" in init_result:
                        raise ConnectionError(
                            f"Failed to initialize: {init_result['error']}"
                        )
                except json.JSONDecodeError as e:
                    self._logger.error(f"Failed to parse JSON response: {e}")
                    raise ConnectionError(
                        f"Failed to parse initialization response: {e}"
                    )
            else:
                # Unexpected content type
                self._logger.warning(f"Unexpected content type: {content_type}")
                raise ConnectionError(
                    f"Unexpected content type during initialization: {content_type}"
                )

            # Check for session ID in headers
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
                self._logger.debug(f"Session established with ID: {session_id}")

            # Send initialized notification
            headers: Dict[str, str] = {}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id

            notification: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {},
            }
            await client.post(
                self.mcp_endpoint()
                if callable(self.mcp_endpoint)
                else self.mcp_endpoint,
                json=notification,
                headers=headers,
            )

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
            self._logger.info("MCP protocol initialized successfully")

        except Exception as e:
            self._logger.error(f"Error connecting to server: {e}")
            self._initialized = False
            raise ConnectionError(f"Could not connect to server {self.server_url}: {e}")
        finally:
            # Always close the client
            await client.aclose()

    async def cleanup_async(self) -> None:
        """
        Clean up the server connection.

        Closes the HTTP client connection if it exists. If a session ID was
        established, attempts to terminate the session with a DELETE request.

        Returns:
            None
        """
        self._logger.info(f"Closing connection with HTTP server: {self.server_url}")

        # If we have a session ID, try to terminate the session
        if self._session_id:
            client = None
            try:
                client = await self._create_client()
                headers = {"Mcp-Session-Id": self._session_id}
                await client.delete(
                    self.mcp_endpoint()
                    if callable(self.mcp_endpoint)
                    else self.mcp_endpoint,
                    headers=headers,
                )
                self._logger.debug(f"Session terminated: {self._session_id}")

                # Remove from session manager
                server_key = self._server_key
                await self.session_manager.delete_session(server_key)

            except Exception as e:
                self._logger.warning(f"Failed to terminate session: {e}")
            finally:
                # Always close the client
                if client:
                    await client.aclose()

        # Close the session manager
        await self.session_manager.close()

        self._session_id = None
        self._last_event_id = None
        self._initialized = False

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
            httpx.RequestError: If there's an error during the HTTP request
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
            httpx.RequestError: If there's an error during the HTTP request
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
            httpx.RequestError: If there's an error during the HTTP request
        """
        from mcp.types import CallToolResult

        response = await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments or {}}
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

                            # Update session data
                            server_key = self._server_key
                            session_data = (
                                await self.session_manager.get_session(server_key) or {}
                            )
                            session_data["last_event_id"] = event_id
                            await self.session_manager.store_session(
                                server_key, session_data
                            )
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
        # Ensure we're connected first
        if not self._initialized:
            self._logger.debug("Server not initialized, connecting first")
            await self.connect_async()

        if not self._initialized:
            raise ConnectionError("Failed to initialize connection")

        # Create a new client for this request (bound to current event loop)
        client = await self._create_client()

        try:
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

            # Prepare headers
            headers: MutableMapping[str, str] = {}
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id

            # Send the request
            response = await client.post(
                self.mcp_endpoint()
                if callable(self.mcp_endpoint)
                else self.mcp_endpoint,
                json=request,
                headers=headers,
            )

            # Check for session ID in response
            if "Mcp-Session-Id" in response.headers and not self._session_id:
                self._session_id = response.headers["Mcp-Session-Id"]
                self._logger.debug(f"Session established with ID: {self._session_id}")

                # Update session ID in session data
                server_key = self._server_key
                session_data = await self.session_manager.get_session(server_key) or {}
                session_data["session_id"] = self._session_id
                await self.session_manager.store_session(server_key, session_data)

            # Handle different response types
            if response.status_code == 404 and self._session_id:
                # Session expired, we need to reconnect
                self._logger.warning("Session expired, reconnecting...")
                self._session_id = None
                self._initialized = False

                # Clear session
                await self.session_manager.delete_session(server_key)

                # Reconnect and retry the request
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
        except ConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            # Handle other exceptions
            self._logger.error(f"Error sending request: {e}")
            raise ConnectionError(f"Error sending request: {e}")
        finally:
            # Always close the client
            await client.aclose()
