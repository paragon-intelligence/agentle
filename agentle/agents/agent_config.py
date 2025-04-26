from typing import Self
from agentle.generations.models.generation.generation_config import GenerationConfig
from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator


class AgentConfig(BaseModel):
    generationConfig: GenerationConfig = Field(default_factory=GenerationConfig)
    maxToolCalls: int = Field(default=15)
    maxIterations: int = Field(default=10)

    @model_validator(mode="after")
    def validate_max_tool_calls(self) -> Self:
        if self.generationConfig.n > 1:
            raise ValueError(
                "a number of choices > 1 is not supported for agents. This is NOT planned to be supported."
                + "If you want multiple choices/responses, just call the agent n times."
            )
        return self
