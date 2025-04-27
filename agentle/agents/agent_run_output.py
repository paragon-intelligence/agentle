import logging
from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.context import Context
from agentle.agents.models.agent_usage_statistics import AgentUsageStatistics
from agentle.agents.models.artifact import Artifact
from agentle.agents.tasks.task_state import TaskState

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    artifacts: Sequence[Artifact]
    task_status: TaskState
    usage: AgentUsageStatistics
    final_context: Context
    parsed: T_StructuredOutput
