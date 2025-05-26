"""
Stdio implementation of the Model Context Protocol (MCP) server client.

This module provides a client implementation for interacting with MCP servers over
standard input/output streams. It enables connection management, tool discovery,
resource querying, and tool execution through stdin/stdout communication.

The implementation follows the MCPServerProtocol interface and uses asyncio streams
for communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from collections.abc import Callable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict, override

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


# TypedDict definitions for JSON-RPC messages
class _JsonRpcRequestParams(TypedDict, total=False):
    """Parameters for a JSON-RPC request."""

    protocolVersion: NotRequired[str]
    clientInfo: NotRequired[MutableMapping[str, str]]
    capabilities: NotRequired[MutableMapping[str, MutableMapping[str, Any]]]
    uri: NotRequired[str]
    tool: NotRequired[str]
    arguments: NotRequired[MutableMapping[str, Any]]


class _JsonRpcRequest(TypedDict):
    """A JSON-RPC request message."""

    jsonrpc: str
    id: str
    method: str
    params: NotRequired[_JsonRpcRequestParams]


class _JsonRpcNotification(TypedDict):
    """A JSON-RPC notification message."""

    jsonrpc: str
    method: str
    params: NotRequired[MutableMapping[str, Any]]


class _JsonRpcResponse(TypedDict, total=False):
    """A JSON-RPC response message."""

    jsonrpc: str
    id: str
    result: NotRequired[MutableMapping[str, Any]]
    error: NotRequired[MutableMapping[str, Any]]


class StdioMCPServer(MCPServerProtocol):
    """
    Stdio implementation of the MCP (Model Context Protocol) server client.

    This class provides a client implementation for interacting with MCP servers
    over standard input/output streams. It handles connection management, tool discovery,
    resource management, and tool invocation through stdin/stdout communication.

    Attributes:
        server_name (str): A human-readable name for the server
        command (str): The command to run the server executable
        server_env (MutableMapping[str, str]): Environment variables to pass to the server process
        working_dir (str): Working directory for the server process

    Usage:
        server = StdioMCPServer(
            server_name="Example MCP",
            command="path/to/server",
            server_env={"DEBUG": "1"},
            working_dir="/path/to/working/dir"
        )
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("tool_name", {"param": "value"})
        await server.cleanup()
    """

    # Required configuration fields
    server_name: str = Field(..., description="Human-readable name for the MCP server")
    command: str | Callable[..., str] = Field(
        ..., description="Command to run the MCP server process"
    )

    # Optional configuration fields
    server_env: MutableMapping[str, str] = Field(
        default_factory=dict,
        description="Environment variables to pass to the server process",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory for the server process",
    )
    request_timeout_s: float = Field(
        default=100.0, description="Timeout in seconds for requests"
    )

    # Internal state
    _process: Optional[asyncio.subprocess.Process] = PrivateAttr(default=None)
    _stdin: Optional[asyncio.StreamWriter] = PrivateAttr(default=None)
    _stdout: Optional[asyncio.StreamReader] = PrivateAttr(default=None)
    _next_id: int = PrivateAttr(default=1)
    _pending_requests: MutableMapping[str, asyncio.Future[_JsonRpcResponse]] = (
        PrivateAttr(default_factory=dict)
    )
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__),
    )
    _read_task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)

    @override
    async def connect_async(self) -> None:
        """
        Connect to the MCP server over stdin/stdout.

        Launches the server process and sets up communication channels.
        Initializes the MCP protocol.

        Raises:
            ConnectionError: If the server process cannot be started
            TimeoutError: If the initialization takes too long
        """
        self._logger.info(f"Starting MCP server process: {self.command}")

        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.server_env)

        try:
            # Split command into args if provided as a string
            cmd_args = shlex.split(
                self.command() if callable(self.command) else self.command
            )

            # Start the server process
            self._process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.working_dir,
            )

            if self._process.stdin is None or self._process.stdout is None:
                raise ConnectionError(
                    "Failed to open stdin/stdout pipes to server process"
                )

            self._stdin = self._process.stdin
            self._stdout = self._process.stdout

            # Start background task to read from stdout
            self._read_task = asyncio.create_task(self._read_responses())

            # Initialize the MCP protocol
            await self._initialize_protocol()

        except (OSError, Exception) as e:
            self._logger.error(f"Failed to start server process: {e}")
            await self.cleanup_async()
            raise ConnectionError(f"Could not start server process: {e}")

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
        Clean up the server connection.

        Terminates the server process and cleans up resources.
        """
        # Cancel the read task if it exists
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        # Close pipes
        if self._stdin is not None:
            self._stdin.close()
            try:
                await self._stdin.wait_closed()
            except Exception:
                pass
            self._stdin = None

        # Terminate the process
        if self._process is not None:
            self._logger.info(f"Terminating MCP server process: {self.command}")
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._logger.warning("Server process did not terminate, killing it")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                # Process already terminated
                pass
            self._process = None

    async def _initialize_protocol(self) -> None:
        """
        Initialize the MCP protocol with the server.

        Sends the initialize request and waits for the response.
        """
        self._logger.info("Initializing MCP protocol")

        initialize_request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "initialize",
            "params": {
                "protocolVersion": "0.9.0",
                "clientInfo": {"name": "agentle-mcp-client", "version": "0.1.0"},
                "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
            },
        }
        self._next_id += 1

        response = await self._send_request(initialize_request)
        if "error" in response:
            raise ConnectionError(
                f"Failed to initialize MCP protocol: {response['error']}"
            )

        # Send initialized notification
        initialized_notification: _JsonRpcNotification = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        }
        await self._send_notification(initialized_notification)

        self._logger.info("MCP protocol initialized successfully")

    async def _read_responses(self) -> None:
        """
        Background task to read responses from the server.

        Continuously reads JSON-RPC messages from stdout and resolves pending requests.
        """
        if self._stdout is None:
            return

        while True:
            try:
                # Read a line from stdout
                line = await self._stdout.readline()
                if not line:
                    self._logger.warning("Server closed stdout")
                    break

                # Parse the JSON-RPC message
                try:
                    message_str = line.decode("utf-8")
                    message = json.loads(message_str)
                    self._logger.debug(f"Received message: {message}")

                    # Handle response to a request
                    if "id" in message:
                        request_id = message["id"]
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            future.set_result(message)
                        else:
                            self._logger.warning(
                                f"Received response for unknown request ID: {request_id}"
                            )

                    # Handle notification or request from server
                    elif "method" in message:
                        # TODO: Implement handling of server requests and notifications if needed
                        self._logger.debug(
                            f"Received notification or request: {message}"
                        )

                except json.JSONDecodeError as e:
                    message_str = line.decode("utf-8", errors="replace")
                    self._logger.error(
                        f"Failed to parse JSON from server: {e}, line: {message_str}"
                    )

            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                self._logger.error(f"Error reading from server: {e}")
                # Continue reading, don't break the loop on errors

    async def _send_request(self, request: _JsonRpcRequest) -> _JsonRpcResponse:
        """
        Send a request to the server and wait for the response.

        Args:
            request (_JsonRpcRequest): The JSON-RPC request to send

        Returns:
            _JsonRpcResponse: The JSON-RPC response

        Raises:
            ConnectionError: If the server is not connected
            TimeoutError: If the request times out
        """
        if self._stdin is None or self._stdout is None:
            raise ConnectionError("Server not connected")

        # Set up a future to receive the response
        request_id = request["id"]
        response_future: asyncio.Future[_JsonRpcResponse] = asyncio.Future()
        self._pending_requests[request_id] = response_future

        # Send the request
        request_json = json.dumps(request) + "\n"
        self._stdin.write(request_json.encode("utf-8"))
        await self._stdin.drain()

        # Wait for the response with timeout
        try:
            return await asyncio.wait_for(
                response_future, timeout=self.request_timeout_s
            )
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(
                f"Request timed out after {self.request_timeout_s} seconds"
            )

    async def _send_notification(self, notification: _JsonRpcNotification) -> None:
        """
        Send a notification to the server.

        Args:
            notification (_JsonRpcNotification): The JSON-RPC notification to send

        Raises:
            ConnectionError: If the server is not connected
        """
        if self._stdin is None:
            raise ConnectionError("Server not connected")

        # Send the notification
        notification_json = json.dumps(notification) + "\n"
        self._stdin.write(notification_json.encode("utf-8"))
        await self._stdin.drain()

    async def list_tools_async(self) -> Sequence[Tool]:
        """
        List the tools available on the server.

        Returns:
            Sequence[Tool]: A list of Tool objects available on the server

        Raises:
            ConnectionError: If the server is not connected
        """
        from mcp.types import Tool

        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "listTools",
            "params": {},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to list tools: {response['error']}")

        if "result" not in response or "tools" not in response["result"]:
            raise ValueError("Invalid response format: missing 'tools' in result")

        return [Tool.model_validate(tool) for tool in response["result"]["tools"]]

    @override
    async def list_resources_async(self) -> Sequence[Resource]:
        """
        List the resources available on the server.

        Returns:
            Sequence[Resource]: A list of Resource objects available on the server

        Raises:
            ConnectionError: If the server is not connected
        """
        from mcp.types import Resource

        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "listResources",
            "params": {},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to list resources: {response['error']}")

        if "result" not in response or "resources" not in response["result"]:
            raise ValueError("Invalid response format: missing 'resources' in result")

        return [
            Resource.model_validate(resource)
            for resource in response["result"]["resources"]
        ]

    @override
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

        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "readResource",
            "params": {"uri": uri},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to read resource contents: {response['error']}")

        if "result" not in response or "contents" not in response["result"]:
            raise ValueError("Invalid response format: missing 'contents' in result")

        return [
            TextResourceContents.model_validate(content)
            if content["type"] == "text"
            else BlobResourceContents.model_validate(content)
            for content in response["result"]["contents"]
        ]

    @override
    async def call_tool_async(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> "CallToolResult":
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

        request: _JsonRpcRequest = {
            "jsonrpc": "2.0",
            "id": str(self._next_id),
            "method": "callTool",
            "params": {"tool": tool_name, "arguments": arguments or {}},
        }
        self._next_id += 1

        response = await self._send_request(request)
        if "error" in response:
            raise ValueError(f"Failed to call tool: {response['error']}")

        if "result" not in response:
            raise ValueError("Invalid response format: missing 'result'")

        return CallToolResult.model_validate(response["result"])
