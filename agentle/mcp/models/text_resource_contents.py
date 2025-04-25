from typing import Literal

from agentle.mcp.models.resource_contents import ResourceContents
from rsb.models.field import Field


class TextResourceContents(ResourceContents):
    """Text contents of a resource."""

    type: Literal["text"] = Field(default="text")

    text: str
    """
    The text of the item. This must only be set if the item can actually be represented
    as text (not binary data).
    """
