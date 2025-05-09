from __future__ import annotations

import json
from typing import cast

from openai.types.chat.chat_completion_message import ChatCompletionMessage
from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)


class OpenAIMessageToGeneratedAssistantMessageAdapter[T](
    Adapter["ChatCompletionMessage", GeneratedAssistantMessage[T]]
):
    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None) -> None:
        super().__init__()
        self.response_schema = response_schema

    def adapt(self, _f: ChatCompletionMessage) -> GeneratedAssistantMessage[T]:
        openai_message = _f
        if openai_message.content is None:
            raise ValueError("Contents of OpenAI message are none. Coudn't proceed.")

        return GeneratedAssistantMessage[T](
            parts=[TextPart(text=openai_message.content)],
            parsed=self.response_schema(**json.loads(openai_message.content))
            if self.response_schema
            else cast(T, None),
        )
