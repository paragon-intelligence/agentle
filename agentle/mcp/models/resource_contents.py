"""
Resource Contents Module for MCP.

This module defines the ResourceContents base class which represents the
contents of a specific resource in the Model Control Protocol (MCP) system.
It provides a common structure for resource metadata that is extended by
more specific resource content types.
"""

from typing import Annotated

from rsb.models.any_url import AnyUrl
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.url_constraints import UrlConstraints


class ResourceContents(BaseModel):
    """
    The base class for the contents of a specific resource in the MCP system.

    ResourceContents provides a common structure for resource metadata, including
    a URI for identification and an optional MIME type to indicate the content format.
    This class serves as a base class for more specific resource content types like
    TextResourceContents and BlobResourceContents.

    Attributes:
        uri (AnyUrl): The URI that uniquely identifies this resource
        mimeType (str | None): The media type of the resource content, if known
    """

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)] = Field(
        ..., description="The URI that uniquely identifies this resource"
    )
    mimeType: str | None = Field(
        default=None, description="The media type of the resource content, if known"
    )
    model_config = ConfigDict(extra="allow")
