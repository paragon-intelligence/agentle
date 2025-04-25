from agentle.generations.models.generation.generation_config import GenerationConfig
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class AgentConfig(BaseModel):
    generationConfig: GenerationConfig = Field(default_factory=GenerationConfig)
    maxToolCalls: int = Field(default=15)
    maxIterations: int = Field(default=10)
