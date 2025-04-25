from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = Field(default=None)
