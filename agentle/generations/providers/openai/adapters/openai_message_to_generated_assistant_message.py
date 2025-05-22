from __future__ import annotations

import json
from collections.abc import MutableSequence
from typing import cast

from openai.types.chat.chat_completion_message import ChatCompletionMessage
from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
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
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )

        openai_message = _f
        if openai_message.content is None:
            raise ValueError("Contents of OpenAI message are none. Coudn't proceed.")

        tool_calls: MutableSequence[ChatCompletionMessageToolCall] = (
            openai_message.tool_calls or []
        )

        tool_parts = [
            ToolExecutionSuggestion(
                tool_name=tool_call.function.name,
                args=json.loads(tool_call.function.arguments or "{}"),
            )
            for tool_call in tool_calls
        ]

        return GeneratedAssistantMessage[T](
            parts=[TextPart(text=openai_message.content)] + tool_parts,
            parsed=self.response_schema(**json.loads(openai_message.content))
            if self.response_schema
            else cast(T, None),
        )
