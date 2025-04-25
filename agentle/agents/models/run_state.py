from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.middleware.response_middleware import ResponseMiddleware
from agentle.agents.tools.agent_tool import AgentTool


class RunState[T_Schema = str](BaseModel):
    task_completed: bool = Field(...)
    iteration: int
    tool_calls_amount: int
    called_tools: set[AgentTool]
    last_response: ResponseMiddleware[T_Schema] | None = None
