from collections.abc import Sequence

from agentle.mcp.models.embedded_resource import EmbeddedResource
from agentle.mcp.models.image_content import ImageContent
from agentle.mcp.models.text_content import TextContent
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class CallToolResult(BaseModel):
    """The server's response to a tool call."""

    metadata: dict[str, object] = Field(default_factory=dict)
    content: Sequence[TextContent | ImageContent | EmbeddedResource]
    isError: bool = Field(default=False)
