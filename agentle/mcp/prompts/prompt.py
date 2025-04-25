from collections.abc import Sequence

from agentle.mcp.prompts.prompt_argument import PromptArgument
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Prompt(BaseModel):
    name: str
    description: str | None = Field(default=None)
    arguments: Sequence[PromptArgument]
