from collections.abc import Sequence
from typing import Annotated

from agentle.mcp.models.role import Role
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class Annotations(BaseModel):
    audience: Sequence[Role] | None = None
    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    model_config = ConfigDict(extra="allow")
