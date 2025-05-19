"""
MCP Server Protocol Module

This module defines the abstract base class for Model Context Protocol servers.
It provides a standardized interface for different server implementations to
connect to external resources, list available tools, and invoke tools.
"""

from __future__ import annotations

import abc
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict

if TYPE_CHECKING:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        Resource,
        TextResourceContents,
        Tool,
    )


class MCPServerProtocol(BaseModel, abc.ABC):
    """
    Abstract base class defining the protocol for MCP servers.

    This class establishes the common interface that all MCP server implementations
    must adhere to, including connection management, tool discovery, resource
    listing, and tool invocation.

    Implementing classes must provide concrete implementations for all abstract
    methods defined in this interface.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get a readable name for the server.

        Returns:
            str: A human-readable name identifying the server.
        """
        ...

    def connect(self) -> None:
        run_sync(self.connect_async)

    def cleanup(self) -> None:
        return run_sync(self.cleanup_async)

    def list_tools(self) -> Sequence[Tool]:
        return run_sync(self.list_tools_async)

    def list_resources(self) -> Sequence[Resource]:
        return run_sync(self.list_resources_async)

    def list_resource_contents(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
        return run_sync(self.list_resource_contents_async, uri=uri)

    def call_tool(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> CallToolResult:
        return run_sync(self.call_tool_async, tool_name=tool_name, arguments=arguments)

    @abc.abstractmethod
    async def connect_async(self) -> None:
        """
        Connect to the MCP server.

        Establishes a connection to the server, which might involve spawning a subprocess,
        opening a network connection, or other initialization steps. The server is expected
        to remain connected until `cleanup()` is called.

        Returns:
            None

        Raises:
            ConnectionError: If connection cannot be established.
        """
        ...

    @abc.abstractmethod
    async def cleanup_async(self) -> None:
        """
        Clean up the server connection.

        Performs necessary cleanup operations such as closing a subprocess,
        terminating a network connection, or releasing other resources.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    async def list_tools_async(self) -> Sequence[Tool]:
        """
        List the tools available on the server.

        Retrieves a list of tools that are available for use through this server.

        Returns:
            Sequence[Tool]: A sequence of Tool objects describing the available tools.

        Raises:
            ConnectionError: If the server is not connected.
        """
        ...

    @abc.abstractmethod
    async def list_resources_async(self) -> Sequence[Resource]:
        """
        List the resources available on the server.

        Retrieves a list of resources that are available through this server.

        Returns:
            Sequence[Resource]: A sequence of Resource objects describing the available resources.

        Raises:
            ConnectionError: If the server is not connected.
        """
        ...

    @abc.abstractmethod
    async def list_resource_contents_async(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
        """
        List the contents of a specific resource.

        Retrieves the contents of a resource identified by its URI.

        Args:
            uri (str): The URI of the resource to retrieve contents for.

        Returns:
            Sequence[TextResourceContents | BlobResourceContents]: A sequence of resource content objects,
            which can be either text or binary data.

        Raises:
            ConnectionError: If the server is not connected.
            ValueError: If the URI is invalid or the resource does not exist.
        """
        ...

    @abc.abstractmethod
    async def call_tool_async(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> CallToolResult:
        """
        Invoke a tool on the server.

        Calls a specified tool with the provided arguments and returns the result.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (MutableMapping[str, object] | None): Arguments to pass to the tool, or None if no arguments.

        Returns:
            CallToolResult: The result of the tool invocation.

        Raises:
            ConnectionError: If the server is not connected.
            ValueError: If the tool does not exist or the arguments are invalid.
        """
        ...
