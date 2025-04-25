import datetime
import uuid
from typing import TYPE_CHECKING

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.cerebras._adapters.cerebras_message_to_generated_assistant_message_adapter import (
    CerebrasMessageToGeneratedAssistantMessageAdapter,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatCompletionResponse


class CerebrasCompletionToGenerationAdapter[T](
    Adapter["ChatCompletionResponse", Generation[T]]
):
    response_schema: type[T] | None
    start_time: datetime.datetime
    model: str
    message_to_generated_assistant_message_adapter: (
        CerebrasMessageToGeneratedAssistantMessageAdapter[T]
    )
    preferred_id: uuid.UUID | None

    def __init__(
        self,
        *,
        response_schema: type[T] | None = None,
        start_time: datetime.datetime,
        model: str,
        message_to_generated_assistant_message_adapter: CerebrasMessageToGeneratedAssistantMessageAdapter[
            T
        ]
        | None = None,
        preferred_id: uuid.UUID | None = None,
    ):
        self.response_schema = response_schema
        self.model = model
        self.start_time = start_time
        self.message_to_generated_assistant_message_adapter = (
            message_to_generated_assistant_message_adapter
            or CerebrasMessageToGeneratedAssistantMessageAdapter(
                response_schema=response_schema
            )
        )
        self.preferred_id = preferred_id

    def adapt(self, _f: ChatCompletionResponse) -> Generation[T]:
        choices: list[Choice[T]] = [
            Choice(
                index=index,
                message=self.message_to_generated_assistant_message_adapter.adapt(
                    choice.message
                ),
            )
            for index, choice in enumerate(_f.choices)
        ]

        usage = Usage(
            prompt_tokens=_f.usage.prompt_tokens or 0,
            completion_tokens=_f.usage.completion_tokens or 0,
        )

        return Generation(
            elapsed_time=datetime.datetime.now() - self.start_time,
            id=self.preferred_id or uuid.uuid4(),
            choices=choices,
            object="chat.generation",
            created=datetime.datetime.now(),
            model=_f.model,
            usage=usage,
        )
