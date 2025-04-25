from typing import Literal

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class TextPart(BaseModel):
    text: str
    type: Literal["text"] = Field(default="text")
