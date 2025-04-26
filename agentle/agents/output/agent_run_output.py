from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.context import Context
from agentle.agents.models.agent_usage_statistics import AgentUsageStatistics
from agentle.agents.models.artifact import Artifact


class AgentRunOutput[T_StructuredOutput](BaseModel):
    artifacts: Sequence[Artifact[T_StructuredOutput]]
    usage: AgentUsageStatistics
    final_context: Context
