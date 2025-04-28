from __future__ import annotations

from collections.abc import MutableSequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.generation.usage import Usage


class RunState[T_Schema = str](BaseModel):
    iteration: int
    tool_calls_amount: int
    last_response: T_Schema | str | None = None
    token_usages: MutableSequence[Usage] = Field(default_factory=list)

    @classmethod
    def init_state(cls) -> RunState[T_Schema]:
        return cls(
            iteration=0,
            tool_calls_amount=0,
            last_response=None,
            token_usages=[],
        )

    def update(
        self,
        last_response: T_Schema | str,
        tool_calls_amount: int,
        iteration: int,
        token_usage: Usage,
    ) -> None:
        self.last_response = last_response
        self.tool_calls_amount = tool_calls_amount
        self.iteration = iteration
        self.token_usages.append(token_usage)
