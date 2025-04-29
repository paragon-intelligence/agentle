"""
Call Tool Result module for MCP.

This module defines the CallToolResult class which represents the response from
a tool invocation in the Model Control Protocol (MCP) system. It provides a
structure for returning various content types (text, images, resources) along with
metadata and error status from tool executions.
"""

from collections.abc import Sequence

from agentle.mcp.models.embedded_resource import EmbeddedResource
from agentle.mcp.models.image_content import ImageContent
from agentle.mcp.models.text_content import TextContent
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class CallToolResult(BaseModel):
    """
    The server's response to a tool call in the MCP system.

    CallToolResult encapsulates the results of a tool execution, providing
    a flexible structure that can include text, images, and embedded resources.
    It also includes metadata about the execution and an error flag to indicate
    whether the tool execution was successful.

    Attributes:
        metadata (dict[str, object]): Additional metadata about the tool execution,
            which can include runtime statistics, version information, or other
            tool-specific details.
        content (Sequence[TextContent | ImageContent | EmbeddedResource]): The main
            content of the tool result, which can be a mix of text, images, and
            embedded resources.
        isError (bool): Flag indicating whether the tool execution resulted in an error.
            When True, the content typically contains error messages.
    """

    metadata: dict[str, object] = Field(
        default_factory=dict, description="Additional metadata about the tool execution"
    )
    content: Sequence[TextContent | ImageContent | EmbeddedResource] = Field(
        ...,
        description="Result content from the tool execution (text, images, resources)",
    )
    isError: bool = Field(
        default=False,
        description="Flag indicating whether an error occurred during tool execution",
    )
