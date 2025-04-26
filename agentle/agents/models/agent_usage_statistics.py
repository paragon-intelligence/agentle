from rsb.models.base_model import BaseModel

from agentle.generations.models.generation.usage import Usage


class AgentUsageStatistics(BaseModel):
    token_usage: Usage
