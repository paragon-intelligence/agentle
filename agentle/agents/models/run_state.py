from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.tasks.task_state import TaskState
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class RunState[T_Schema = str](BaseModel):
    iteration: int
    tool_calls_amount: int
    called_tools: dict[ToolExecutionSuggestion, Any] = Field(
        description="A dictionary of tool execution suggestions and their results (tool calls)"
    )
    last_response: T_Schema | str | None = None
    token_usages: MutableSequence[Usage] = Field(default_factory=list)
    task_status: TaskState = Field(default=TaskState.SUBMITTED)

    @classmethod
    def init_state(cls) -> RunState[T_Schema]:
        return cls(
            iteration=0,
            tool_calls_amount=0,
            called_tools={},
            last_response=None,
            token_usages=[],
            task_status=TaskState.SUBMITTED,
        )

    def update(
        self,
        last_response: T_Schema | str,
        called_tools: dict[ToolExecutionSuggestion, Any],
        tool_calls_amount: int,
        iteration: int,
        token_usage: Usage,
        task_status: TaskState,
    ) -> None:
        self.last_response = last_response
        self.called_tools = called_tools
        self.tool_calls_amount = tool_calls_amount
        self.iteration = iteration
        self.token_usages.append(token_usage)
        self.task_status = task_status
