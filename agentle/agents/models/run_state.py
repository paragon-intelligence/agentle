from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.middleware.response_middleware import ResponseMiddleware
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class RunState[T_Schema = str](BaseModel):
    task_completed: bool = Field(...)
    iteration: int
    tool_calls_amount: int
    called_tools: set[ToolExecutionSuggestion]
    last_response: ResponseMiddleware[T_Schema] | ResponseMiddleware[str] | None = None

    def update(
        self,
        task_completed: bool,
        last_response: ResponseMiddleware[T_Schema] | ResponseMiddleware[str],
        called_tools: set[ToolExecutionSuggestion],
        tool_calls_amount: int,
        iteration: int,
    ) -> None:
        self.task_completed = task_completed
        self.last_response = last_response
        self.called_tools = called_tools
        self.tool_calls_amount = tool_calls_amount
        self.iteration = iteration
