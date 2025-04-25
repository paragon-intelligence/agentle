from typing import Literal

from rsb.models.base64str import Base64Str
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ImageMessageContent(BaseModel):
    data: Base64Str | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    type: Literal["image"] = Field(default="image")
