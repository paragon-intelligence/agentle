from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class DataPart(BaseModel):
    type: Literal["data"] = Field(default="data")
    data: dict[str, Any]
