"""
Resource Description With Template Module for MCP.

This module defines the ResourceDescriptionWithTemplate class which represents
metadata about a templated resource in the Model Control Protocol (MCP) system.
It provides a structure for describing resources that can be instantiated with
parameters through a URI template.
"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ResourceDescriptionWithTemplate(BaseModel):
    """
    Metadata describing a templated resource in the MCP system.

    ResourceDescriptionWithTemplate provides a structure for parameterized resource
    metadata, including a URI template, name, and optional description and media type.
    This class is used for describing resources that can be instantiated with
    different parameters, enabling dynamic resource generation within the MCP ecosystem.

    Attributes:
        uriTemplate (str): Template pattern for generating resource URIs with parameters
        name (str): A human-readable name for the resource
        description (str | None): An optional human-readable description of the resource
        mimeType (str | None): The media type of the resource content
    """

    uriTemplate: str = Field(
        ..., description="Template pattern for generating resource URIs with parameters"
    )
    name: str = Field(..., description="A human-readable name for the resource")
    description: str | None = Field(
        default=None, description="A human-readable description of the resource"
    )
    mimeType: str | None = Field(
        default=None, description="The media type of the resource content"
    )
