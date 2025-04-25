from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class InputSchema(BaseModel):
    properties: dict[str, object] = Field(description="Tool specific parameters")
    type: Literal["object"] = Field(default="object")
