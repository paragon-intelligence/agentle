from __future__ import annotations

import datetime
from typing import TYPE_CHECKING
import uuid

from openai.types.chat.chat_completion import ChatCompletion
from rsb.adapters.adapter import Adapter


from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.openai.adapters.openai_choice_to_choice_adapter import (
    OpenaiChoiceToChoiceAdapter,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion


class ChatCompletionToGenerationAdapter[T](Adapter["ChatCompletion", Generation[T]]):
    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None) -> None:
        super().__init__()
        self.response_schema = response_schema

    def adapt(self, _f: ChatCompletion) -> Generation[T]:
        from openai.types.completion_usage import CompletionUsage

        completion = _f
        choice_adapter = OpenaiChoiceToChoiceAdapter[T](
            response_schema=self.response_schema
        )

        usage = completion.usage or CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )

        return Generation(
            id=uuid.UUID(completion.id),
            object="chat.generation",
            created=datetime.datetime.fromtimestamp(completion.created),
            model=completion.model,
            choices=[choice_adapter.adapt(choice) for choice in completion.choices],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
        )
