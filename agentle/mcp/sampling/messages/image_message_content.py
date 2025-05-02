"""
Image Message Content module for MCP.

This module defines the ImageMessageContent class, which represents image-based
content in messages within the Model Control Protocol (MCP) system. It provides
a structure for including image data in Base64 format within messages.
"""

from typing import Literal

from rsb.models.base64str import Base64Str
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ImageMessageContent(BaseModel):
    """
    Image content for messages in the MCP system.

    ImageMessageContent represents the content of a message that contains an image.
    The image data is stored in Base64 format with an associated MIME type to specify
    the image format. The "type" field is always set to "image" to identify this
    content type.

    Attributes:
        data (Base64Str | None): The Base64-encoded image data
        mime_type (str | None): The MIME type of the image, e.g., "image/jpeg"
        type (Literal["image"]): The content type, always set to "image"
    """

    data: Base64Str | None = Field(
        default=None, description="Base64-encoded image data"
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type of the image (e.g., 'image/jpeg', 'image/png')",
    )
    type: Literal["image"] = Field(
        default="image",
        description="Content type identifier, always 'image' for this class",
    )

    class Config:
        arbitrary_types_allowed = True
