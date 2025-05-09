"""
Adapter module for converting Google AI Content objects to Agentle GeneratedAssistantMessage objects.

This module provides the GoogleContentToGeneratedAssistantMessageAdapter class, which
transforms the content objects returned by Google's Generative AI APIs into
the standardized GeneratedAssistantMessage format used throughout the Agentle framework.

The adapter handles the conversion of Google-specific message structures, roles, and
parts into Agentle's unified message representation, ensuring that responses from
Google models can be seamlessly integrated with the rest of the framework.

This adapter is typically used internally by the GoogleGenerationProvider to process
raw responses from Google's API before returning them to the application code.

Example:
```python
from agentle.generations.providers.google._adapters.google_content_to_generated_assistant_message_adapter import (
    GoogleContentToGeneratedAssistantMessageAdapter
)
from google.genai.types import Content, Part

# Create a Google Content object (typically received from API)
google_part = Part(text="Hello, I'm a Google AI assistant.")
google_content = Content(role="model", parts=[google_part])

# Use the adapter to convert to Agentle's format
adapter = GoogleContentToGeneratedAssistantMessageAdapter()
agentle_message = adapter.adapt(google_content)

# Now use the standardized message in your application
print(agentle_message.parts[0].text)  # "Hello, I'm a Google AI assistant."
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.providers.google.adapters.google_part_to_part_adapter import (
    GooglePartToPartAdapter,
)

if TYPE_CHECKING:
    from google.genai.types import Content


class GoogleContentToGeneratedAssistantMessageAdapter[T](
    Adapter["Content", GeneratedAssistantMessage[T]]
):
    """
    Adapter for converting Google AI Content objects to Agentle GeneratedAssistantMessage objects.

    This adapter transforms the Content objects returned by Google's Generative AI APIs
    into the standardized GeneratedAssistantMessage format used within Agentle. It handles
    the mapping of Google's message structure, role validation, and part conversion.

    The adapter is generic over type T, which represents the optional structured data
    that may be parsed from the model response when using structured output schemas.

    This class is part of Agentle's provider abstraction layer, which allows the framework
    to work with multiple AI providers while maintaining a consistent internal API.

    Attributes:
        part_adapter (GooglePartToPartAdapter): Adapter used to convert individual
            Google Part objects to Agentle Part objects. This handles the conversion
            of content parts like text, images, and function calls.

        generate_content_response_parsed (T | None): Optional parsed structured data
            extracted from the model response. This is used when the generation was
            configured to parse structured output according to a schema.

    Example:
        ```python
        # Create an adapter with parsed structured output
        class WeatherInfo:
            location: str
            temperature: float

        parsed_data = WeatherInfo(location="London", temperature=15.5)
        adapter = GoogleContentToGeneratedAssistantMessageAdapter[WeatherInfo](
            generate_content_response_parsed=parsed_data
        )

        # Convert a Google Content object
        agentle_message = adapter.adapt(google_content)

        # Access the parsed structured data
        if agentle_message.parsed:
            print(f"Weather in {agentle_message.parsed.location}: {agentle_message.parsed.temperature}°C")
        ```
    """

    part_adapter: GooglePartToPartAdapter
    generate_content_response_parsed: T | None

    def __init__(
        self,
        part_adapter: GooglePartToPartAdapter | None = None,
        generate_content_response_parsed: T | None = None,
    ) -> None:
        """
        Initialize the adapter with optional part adapter and parsed response.

        Args:
            part_adapter: Optional adapter for converting Google Part objects to Agentle
                Part objects. If not provided, a new GooglePartToPartAdapter is created.
            generate_content_response_parsed: Optional parsed structured data that
                has been extracted from the model response. This is used when the
                generation was configured to parse responses according to a schema.
        """
        super().__init__()
        self.part_adapter = part_adapter or GooglePartToPartAdapter()
        self.generate_content_response_parsed = generate_content_response_parsed

    @override
    def adapt(self, _f: Content) -> GeneratedAssistantMessage[T]:
        """
        Convert a Google Content object to an Agentle GeneratedAssistantMessage object.

        This method extracts parts from the Google Content object, adapts them using
        the part_adapter, validates the role, and constructs a GeneratedAssistantMessage
        with the converted parts and any parsed structured data.

        Args:
            _f: The Google Content object to adapt, typically received from a
                Google AI model response.

        Returns:
            GeneratedAssistantMessage[T]: An Agentle message containing the adapted
                parts and any parsed structured data.

        Raises:
            ValueError: If the Content object has no parts, no role, or a role other
                than "model" (which is what Google uses for assistant/model messages).

        Example:
            ```python
            from google.genai.types import Content, Part

            # Create a Google Content object (typically received from API)
            google_part = Part(text="The weather is sunny.")
            google_content = Content(role="model", parts=[google_part])

            # Adapt it to Agentle's format
            adapter = GoogleContentToGeneratedAssistantMessageAdapter()
            agentle_message = adapter.adapt(google_content)
            ```
        """
        parts = _f.parts

        if parts is None:
            raise ValueError(
                f"No parts found in Google Content. This is a Google bug, please report it. Content: {_f}"
            )

        adapted_parts = [self.part_adapter.adapt(part) for part in parts]

        role = _f.role
        if role is None:
            raise ValueError("No role found in Google Content.")

        match role:
            case "model":
                # Filter parts and explicitly cast the result to the expected type
                filtered_parts: list[TextPart | ToolExecutionSuggestion] = [
                    part
                    for part in adapted_parts
                    if isinstance(part, (TextPart, ToolExecutionSuggestion))
                ]  # GeneratedAssistantMessage only supports TextPart and ToolExecutionSuggestion

                message = GeneratedAssistantMessage[T](
                    parts=filtered_parts,
                    parsed=self.generate_content_response_parsed
                    if self.generate_content_response_parsed
                    else cast(T, None),
                )

                return message
            case _:
                raise ValueError(
                    f"This adapter does only supports assistant messages. Provided: {_f}"
                )
