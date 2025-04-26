from __future__ import annotations

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class Usage(BaseModel):
    prompt_tokens: int = Field(description="Number of tokens consumed by the prompt")
    completion_tokens: int = Field(
        description="Number of tokens consumed by the completion."
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __radd__(self, other: int) -> Usage:
        if other == 0:
            return self
        return Usage(
            prompt_tokens=self.prompt_tokens + other,
            completion_tokens=self.completion_tokens,
        )

    @staticmethod
    def zero() -> Usage:
        return Usage(prompt_tokens=0, completion_tokens=0)
