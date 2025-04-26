from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from rsb.decorators.entities import entity

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)

logger = logging.getLogger(__name__)


@entity
class Generation[T](BaseModel):
    elapsed_time: timedelta
    id: uuid.UUID
    object: Literal["chat.generation"]
    created: datetime
    model: str
    choices: Sequence[Choice[T]]
    usage: Usage

    @property
    def parsed(self) -> T:
        if len(self.choices) > 1:
            raise ValueError(
                "Choices list is > 1. Coudn't determine the parsed "
                + "model to obtain. Please, use the get_parsed "
                + "method, instead, passing the choice number "
                + "you want to get the parsed model."
            )

        return self.get_parsed(0)

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
        if len(self.choices) > 1:
            logger.warning(
                "Choices list is > 1. Coudn't determine the tool calls. "
                + "Please, use the get_tool_calls method, instead, "
                + "passing the choice number you want to get the tool calls."
                + "Returning the first choice tool calls."
            )

        return self.get_tool_calls(0)

    @property
    def tool_calls_amount(self) -> int:
        return len(self.tool_calls)

    def get_tool_calls(self, choice: int = 0) -> Sequence[ToolExecutionSuggestion]:
        return self.choices[choice].message.tool_calls

    @classmethod
    def mock(cls) -> Generation[T]:
        return cls(
            model="mock-model",
            elapsed_time=timedelta(seconds=0),
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.now(),
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    @property
    def text(self) -> str:
        return "".join([choice.message.text for choice in self.choices])

    def get_parsed(self, choice: int) -> T:
        return self.choices[choice].message.parsed
