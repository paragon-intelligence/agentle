from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TextMessageContent(BaseModel):
    text: str | None = Field(description="text of the message")
    type: Literal["text"] = Field(default="text")
