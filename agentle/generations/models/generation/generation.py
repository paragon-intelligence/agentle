from __future__ import annotations
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.usage import Usage
from rsb.decorators.entities import entity


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
