from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.json_rpc_error import JSONRPCError


class JSONRPCResponse[R_Result = dict[str, Any]](BaseModel):
    id: str
    result: R_Result | None = Field(default=None)
    error: JSONRPCError | None = Field(default=None)
