"""
Resource Description Module for MCP.

This module defines the ResourceDescription class which represents metadata about
a resource in the Model Control Protocol (MCP) system. It provides a structure
for describing resources with their identifiers, names, and optional details.
"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ResourceDescription(BaseModel):
    """
    Metadata describing a resource in the MCP system.

    ResourceDescription provides a structure for resource metadata, including
    its identifier, name, and optional description and media type. This class
    is used for resource discovery and management within the MCP ecosystem.

    Attributes:
        uri (str): The unique identifier or location of the resource
        name (str): A human-readable name for the resource
        description (str | None): An optional human-readable description of the resource
        mimeType (str | None): The media type of the resource content
    """

    uri: str = Field(
        ..., description="The unique identifier or location of the resource"
    )
    name: str = Field(..., description="A human-readable name for the resource")
    description: str | None = Field(
        default=None, description="A human-readable description of the resource"
    )
    mimeType: str | None = Field(
        default=None, description="The media type of the resource content"
    )
