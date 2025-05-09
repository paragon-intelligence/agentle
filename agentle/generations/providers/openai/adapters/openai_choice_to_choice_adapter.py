from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.providers.openai.adapters.openai_message_to_generated_assistant_message import (
    OpenAIMessageToGeneratedAssistantMessageAdapter,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import Choice as OpenAIChoice


class OpenaiChoiceToChoiceAdapter[T](Adapter["OpenAIChoice", Choice[T]]):
    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None) -> None:
        super().__init__()
        self.response_schema = response_schema

    def adapt(self, _f: OpenAIChoice) -> Choice[T]:
        openai_choice = _f
        return Choice(
            index=openai_choice.index,
            message=OpenAIMessageToGeneratedAssistantMessageAdapter[T](
                response_schema=self.response_schema
            ).adapt(openai_choice.message),
        )
