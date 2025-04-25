from typing import Sequence

from agentle.mcp.sampling.hint import Hint
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ModelPreference(BaseModel):
    hints: Sequence[Hint] | None = Field(default=None)
    costPriority: float | None = Field(default=None, ge=0.0, le=1.0)
    speedPriority: float | None = Field(default=None, ge=0.0, le=1.0)
    intelligencePriority: float | None = Field(default=None, ge=0.0, le=1.0)
