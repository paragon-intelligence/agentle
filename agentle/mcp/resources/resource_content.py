"""
Resource Content Module for MCP.

This module defines the ResourceContent class which represents the content of
a resource in the Model Control Protocol (MCP) system. It provides a structure
for representing both text and binary data resources with their associated metadata.
"""

from rsb.models.base64str import Base64Str
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ResourceContent(BaseModel):
    """
    Represents the content of a resource in the MCP system.

    ResourceContent provides a structure for resource data, supporting both text
    and binary representations along with metadata like URI and MIME type. This class
    is used for transferring resource contents within the MCP ecosystem.

    Attributes:
        uri (str): The unique identifier or location of the resource
        mimeType (str | None): The media type of the resource content, if known
        text (str | None): The text content if the resource is textual
        blob (Base64Str | None): Base64-encoded data if the resource is binary
    """

    uri: str = Field(
        ..., description="The unique identifier or location of the resource"
    )
    mimeType: str | None = Field(
        default=None, description="The media type of the resource content"
    )
    text: str | None = Field(
        default=None, description="The text content of the resource"
    )
    blob: Base64Str | None = Field(
        default=None, description="Base64-encoded binary data of the resource"
    )

    class Config:
        arbitrary_types_allowed = True
