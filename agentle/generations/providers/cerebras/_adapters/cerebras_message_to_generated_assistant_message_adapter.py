"""
Adapter for converting Cerebras message responses to Agentle's GeneratedAssistantMessage format.

This module provides the CerebrasMessageToGeneratedAssistantMessageAdapter class, which transforms
response messages from Cerebras's API (ChatCompletionResponseChoiceMessage) into Agentle's
internal GeneratedAssistantMessage format. This adapter also supports handling structured
output parsing when a response schema is provided.

This adapter is a key component in the response processing pipeline for the Cerebras
provider implementation, ensuring that responses are normalized to Agentle's standard
format regardless of the underlying provider.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
import json
from typing import TYPE_CHECKING, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)

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
    """
    Adapter for converting Cerebras message responses to Agentle's GeneratedAssistantMessage format.

    This class transforms response messages from Cerebras's API into Agentle's internal
    GeneratedAssistantMessage format. The adapter is generic over type T, which represents
    the optional structured data format that can be extracted from the model's response
    when a response schema is provided.

    Attributes:
        response_schema: Optional Pydantic model class for parsing structured data from
            the response. When provided, the adapter will attempt to extract typed data
            according to this schema.
    """

    response_schema: type[T] | None

    def __init__(self, response_schema: type[T] | None = None):
        """
        Initialize the adapter with an optional response schema.

        Args:
            response_schema: Optional Pydantic model class for parsing structured data
                from the response.
        """
        self.response_schema = response_schema

    def adapt(
        self,
        _f: ChatCompletionResponseChoiceMessage,
    ) -> GeneratedAssistantMessage[T]:
        """
        Convert a Cerebras message response to an Agentle GeneratedAssistantMessage.

        This method transforms a message from Cerebras's response format into Agentle's
        standardized GeneratedAssistantMessage format. If a response schema was provided,
        it will also attempt to parse structured data from the response.

        Args:
            _f: The Cerebras message response to convert.

        Returns:
            GeneratedAssistantMessage[T]: The converted message in Agentle's format,
                potentially with structured output data if a response_schema was provided.
        """
        # Implementation would extract the content from the Cerebras message
        # and create a GeneratedAssistantMessage, potentially with structured data
        from cerebras.cloud.sdk.types.chat.chat_completion import (
            ChatCompletionResponseChoiceMessageToolCall,
        )

        from agentle.generations.models.message_parts.text import TextPart

        # The structured data would be None unless a response schema was provided
        # and the response contained valid JSON matching that schema
        content = _f.content or ""
        tool_calls: Sequence[ChatCompletionResponseChoiceMessageToolCall] | None = (
            _f.tool_calls
        )

        tool_call_parts: MutableSequence[ToolExecutionSuggestion] = [
            ToolExecutionSuggestion(
                tool_name=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
            )
            for tool_call in (tool_calls or [])
        ]

        return GeneratedAssistantMessage[T](
            parts=[TextPart(text=content)] + tool_call_parts,
            parsed=self.response_schema(**json.loads(content))
            if self.response_schema
            else cast(T, None),
        )
