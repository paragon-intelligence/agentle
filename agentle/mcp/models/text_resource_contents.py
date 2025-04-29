"""
Text Resource Contents Module for MCP.

This module defines the TextResourceContents class which represents textual
content for resources in the Model Control Protocol (MCP) system. It provides
a structure for representing resources that can be expressed as text.
"""

from typing import Literal

from agentle.mcp.models.resource_contents import ResourceContents
from rsb.models.field import Field


class TextResourceContents(ResourceContents):
    """
    Text contents of a resource in the MCP system.

    TextResourceContents extends ResourceContents to provide a structure for
    representing textual data as part of resources. This class is used for
    resources that can be represented as text, such as source code files,
    configuration files, log files, JSON/XML data, and plain text documents.

    Attributes:
        type (Literal["text"]): The resource content type, always set to "text"
        text (str): The textual content of the resource

    Inherited Attributes:
        uri (AnyUrl): The URI of this resource
        mimeType (str | None): The MIME type of this resource, if known
    """

    type: Literal["text"] = Field(
        default="text",
        description="Content type identifier, always 'text' for textual resource contents",
    )

    text: str = Field(..., description="The textual content of the resource")
