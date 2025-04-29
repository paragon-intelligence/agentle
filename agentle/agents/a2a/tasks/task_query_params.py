from typing import Any
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TaskQueryParams(BaseModel):
    id: str
    historyLength: int | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)
