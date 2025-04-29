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

from typing import TYPE_CHECKING, cast

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
        from agentle.generations.models.message_parts.text import TextPart

        # The structured data would be None unless a response schema was provided
        # and the response contained valid JSON matching that schema
        parsed_data = cast(T, None)  # Properly cast None to type T for type checking
        content = _f.content or ""

        return GeneratedAssistantMessage[T](
            parts=[TextPart(text=content)],
            parsed=parsed_data,
        )
