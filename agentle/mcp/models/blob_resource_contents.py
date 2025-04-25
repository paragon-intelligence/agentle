from typing import Literal

from agentle.mcp.models.resource_contents import ResourceContents
from rsb.models.field import Field


class BlobResourceContents(ResourceContents):
    """Binary contents of a resource."""

    type: Literal["blob"] = Field(default="blob")

    blob: str
    """A base64-encoded string representing the binary data of the item."""
