"""
Blob Resource Contents module for MCP.

This module defines the BlobResourceContents class which represents binary data
content for resources in the Model Control Protocol (MCP) system. This class is used
for binary files, images, or any non-text resource content that needs to be
represented in base64-encoded format.
"""

from typing import Literal

from agentle.mcp.models.resource_contents import ResourceContents
from rsb.models.field import Field


class BlobResourceContents(ResourceContents):
    """
    Binary contents of a resource in the MCP system.

    BlobResourceContents extends ResourceContents to provide a structure for
    representing binary data as part of resources. The binary data is stored
    as a base64-encoded string. This class is used for any non-textual content
    such as images, PDFs, audio files, or other binary formats.

    Attributes:
        type (Literal["blob"]): The resource content type, always set to "blob"
        blob (str): A base64-encoded string representing the binary data

    Inherited Attributes:
        uri (AnyUrl): The URI of this resource
        mimeType (str | None): The MIME type of this resource, if known
    """

    type: Literal["blob"] = Field(
        default="blob",
        description="Content type identifier, always 'blob' for binary resource contents",
    )

    blob: str = Field(
        ...,
        description="Base64-encoded string representing the binary data of the resource",
    )
