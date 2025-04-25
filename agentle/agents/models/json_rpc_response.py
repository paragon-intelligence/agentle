from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.json_rpc_error import JSONRPCError


class JSONRPCResponse(BaseModel):
    id: str
    result: dict[str, Any] | None = Field(default=None)
    error: JSONRPCError | None = Field(default=None)

