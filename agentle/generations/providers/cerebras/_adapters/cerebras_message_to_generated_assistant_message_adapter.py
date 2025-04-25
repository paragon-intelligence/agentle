from typing import TYPE_CHECKING

from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from cerebras.cloud.sdk.types.chat.chat_completion import (
        ChatCompletionResponseChoiceMessage,
    )


class CerebrasMessageToGeneratedAssistantMessageAdapter[T](
    Adapter[
        "ChatCompletionResponseChoiceMessage",
        GeneratedAssistantMessage[T],
    ]
):
    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None = None):
        self.response_schema = response_schema

    def adapt(
        self,
        _f: ChatCompletionResponseChoiceMessage,
    ) -> GeneratedAssistantMessage[T]: ...
