from typing import Literal

from agentle.mcp.models.annotations import Annotations
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"] = Field(default="text")
    text: str
    """The text content of the message."""
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")
