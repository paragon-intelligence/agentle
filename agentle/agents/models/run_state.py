from __future__ import annotations
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class RunState[T_Schema = str](BaseModel):
    task_completed: bool = Field(...)
    iteration: int
    tool_calls_amount: int
    called_tools: dict[ToolExecutionSuggestion, Any] = Field(
        description="A dictionary of tool execution suggestions and their results (tool calls)"
    )
    last_response: T_Schema | str | None = None

    @classmethod
    def init_state(cls) -> RunState[T_Schema]:
        return cls(
            task_completed=False,
            iteration=0,
            tool_calls_amount=0,
            called_tools={},
            last_response=None,
        )

    def update(
        self,
        task_completed: bool,
        last_response: T_Schema | str,
        called_tools: dict[ToolExecutionSuggestion, Any],
        tool_calls_amount: int,
        iteration: int,
    ) -> None:
        self.task_completed = task_completed
        self.last_response = last_response
        self.called_tools = called_tools
        self.tool_calls_amount = tool_calls_amount
        self.iteration = iteration
