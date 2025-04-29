"""
Image Content Module for MCP.

This module defines the ImageContent class which represents image content
in the Model Control Protocol (MCP) system. It provides a structure for
including base64-encoded images within messages or responses.
"""

from typing import Literal

from agentle.mcp.models.annotations import Annotations
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class ImageContent(BaseModel):
    """
    Image content for a message in the MCP system.

    ImageContent provides a structure for representing image data within
    messages or responses. The image is stored as base64-encoded data with
    an associated MIME type to indicate the image format. Optional annotations
    can be included to provide additional context or metadata about the image.

    Attributes:
        data (str): The base64-encoded image data
        mimeType (str): The MIME type of the image (e.g., "image/jpeg", "image/png")
        annotations (Annotations | None): Optional annotations for this image
        type (Literal["image"]): The content type identifier, always "image"
    """

    data: str = Field(..., description="The base64-encoded image data")
    mimeType: str = Field(
        ..., description="The MIME type of the image (e.g., 'image/jpeg', 'image/png')"
    )
    annotations: Annotations | None = Field(
        default=None, description="Optional annotations for this image"
    )
    model_config = ConfigDict(extra="allow")

    type: Literal["image"] = Field(
        default="image", description="Content type identifier, always 'image'"
    )
