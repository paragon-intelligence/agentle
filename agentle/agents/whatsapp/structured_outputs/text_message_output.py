from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TextMessageOutput(BaseModel):
    type: Literal["text"] = Field(default="text")
    text: str
