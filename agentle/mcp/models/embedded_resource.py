"""
Embedded Resource Module for MCP.

This module defines the EmbeddedResource class which represents a resource
embedded into a prompt or tool call result in the Model Control Protocol (MCP) system.
It provides a structure for including resource contents directly within messages
or responses.
"""

from typing import Literal

from agentle.mcp.models.annotations import Annotations
from agentle.mcp.models.blob_resource_contents import BlobResourceContents
from agentle.mcp.models.text_resource_contents import TextResourceContents
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class EmbeddedResource(BaseModel):
    """
    The contents of a resource, embedded into a prompt or tool call result.

    EmbeddedResource provides a structure for including resource contents directly
    within messages or responses in the MCP system. This allows resources to be
    presented inline rather than requiring separate retrieval. The embedded resource
    can contain either text or binary data, along with optional annotations.

    It is up to the client how best to render embedded resources for the benefit
    of the LLM and/or the user.

    Attributes:
        type (Literal["resource"]): The content type identifier, always "resource"
        resource (TextResourceContents | BlobResourceContents): The actual resource contents,
            either text or binary data
        annotations (Annotations | None): Optional annotations for this resource
    """

    type: Literal["resource"] = Field(
        default="resource", description="Content type identifier, always 'resource'"
    )
    resource: TextResourceContents | BlobResourceContents = Field(
        ...,
        description="The actual resource contents, either text or binary data",
    )
    annotations: Annotations | None = Field(
        default=None, description="Optional annotations for this resource"
    )
    model_config = ConfigDict(extra="allow")
