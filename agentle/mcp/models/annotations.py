"""
MCP Annotations module.

This module defines the Annotations class which provides metadata for content
in the Model Control Protocol (MCP) system. Annotations can be attached to
various content types to provide additional context about audience targeting,
priority, and other extensible metadata.
"""

from collections.abc import Sequence
from typing import Annotated

from agentle.mcp.models.role import Role
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class Annotations(BaseModel):
    """
    Metadata annotations for MCP content.

    The Annotations class provides a way to attach metadata to content items in the
    MCP system. This metadata can be used to specify the intended audience (user,
    assistant, or both), prioritize content display, and store custom attributes.

    Attributes:
        audience (Sequence[Role] | None): The intended recipients of the content.
            Can be used to target content to specific participants in a conversation.
        priority (float | None): A value between 0.0 and 1.0 indicating the display
            priority of the content. Higher values indicate higher priority.

    Note:
        This model allows additional fields beyond those explicitly defined through
        the use of ConfigDict(extra="allow").
    """

    audience: Sequence[Role] | None = Field(
        default=None,
        description="The intended recipients of the content (user, assistant, or both)",
    )
    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Display priority value between 0.0 and 1.0 (higher is more important)",
    )
    model_config = ConfigDict(extra="allow")
