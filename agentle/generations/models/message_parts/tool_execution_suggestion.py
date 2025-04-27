import uuid
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.config_dict import ConfigDict


class ToolExecutionSuggestion(BaseModel):
    type: Literal["tool_execution"] = Field(
        default="tool_execution",
    )
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    args: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @property
    def text(self) -> str:
        return f"Tool: {self.tool_name}\nArgs: {self.args}"
