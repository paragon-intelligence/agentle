import logging
from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.context import Context
from agentle.agents.models.agent_usage_statistics import AgentUsageStatistics
from agentle.agents.models.artifact import Artifact

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    artifacts: Sequence[Artifact[T_StructuredOutput]]
    usage: AgentUsageStatistics
    final_context: Context

    @property
    def parsed(self) -> T_StructuredOutput:
        structured_outputs: list[T_StructuredOutput] = [
            artifact.parsed
            for artifact in reversed(self.artifacts)
            if artifact.parsed is not None
        ]
        if len(structured_outputs) == 0:
            raise ValueError("No parsed artifact found")
        if len(structured_outputs) > 1:
            logger.warning("Multiple parsed artifacts found. Returning the last one.")

        return structured_outputs[-1]
