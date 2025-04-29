"""
Adapter module for converting Google AI Part objects to Agentle Part objects.

This module provides the GooglePartToPartAdapter class, which transforms content parts
from Google's Generative AI API responses into the standardized Part objects used
throughout the Agentle framework. It handles various content types including text,
inline data (like images), and function calls.

The adapter is responsible for ensuring that provider-specific content formats
from Google are converted to Agentle's unified message part representation, maintaining
consistency across different providers while preserving all necessary information.

Example:
```python
from agentle.generations.providers.google._adapters.google_part_to_part_adapter import (
    GooglePartToPartAdapter
)
from google.genai.types import Part as GooglePart

# Create a Google text part
google_text_part = GooglePart(text="Hello, world!")

# Create an adapter
adapter = GooglePartToPartAdapter()

# Convert to Agentle's format
agentle_part = adapter.adapt(google_text_part)

# Now use the standardized part in your application
print(agentle_part.text)  # "Hello, world!"
```
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Never

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.part import Part
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)

if TYPE_CHECKING:
    from google.genai.types import Part as GooglePart


class GooglePartToPartAdapter(Adapter["GooglePart", Part]):
    """
    Adapter for converting Google AI Part objects to Agentle Part objects.

    This adapter transforms the content parts returned by Google's Generative AI API
    into the standardized Part formats used within Agentle. It handles multiple part
    types including:

    - Text content (converted to TextPart)
    - Inline data like images (converted to FilePart)
    - Function calls (converted to ToolExecutionSuggestion)

    The adapter is a key component of Agentle's provider abstraction layer,
    ensuring that all parts of Google's responses can be seamlessly integrated
    with the rest of the framework.

    Example:
        ```python
        # Convert a Google text part
        google_text = GooglePart(text="The answer is 42.")
        adapter = GooglePartToPartAdapter()
        text_part = adapter.adapt(google_text)

        # Convert a Google function call part
        google_function_call = GooglePart(
            function_call={
                "name": "get_weather",
                "args": {"location": "London"},
                "id": "call-123"
            }
        )
        tool_suggestion = adapter.adapt(google_function_call)
        ```
    """

    def adapt(self, _f: "GooglePart") -> Part:
        """
        Convert a Google Part object to an Agentle Part object.

        This method examines the input Google Part to determine its type
        (text, inline data, or function call) and converts it to the appropriate
        Agentle Part subclass.

        Args:
            _f: The Google Part object to adapt, typically received from a
                Google AI model response.

        Returns:
            Part: An Agentle Part object (TextPart, FilePart, or ToolExecutionSuggestion)
                containing the adapted content.

        Raises:
            ValueError: If the part contains invalid or missing required data,
                or if the part type is not supported.

        Example:
            ```python
            # Convert text
            google_part = GooglePart(text="Hello!")
            text_part = adapter.adapt(google_part)  # Returns TextPart

            # Convert image data
            google_part = GooglePart(
                inline_data={
                    "mime_type": "image/jpeg",
                    "data": b"<binary image data>"
                }
            )
            file_part = adapter.adapt(google_part)  # Returns FilePart

            # Convert function call
            google_part = GooglePart(
                function_call={
                    "name": "calculate",
                    "args": {"expression": "2+2"},
                    "id": "func-1"
                }
            )
            tool_part = adapter.adapt(google_part)  # Returns ToolExecutionSuggestion
            ```
        """
        if _f.text:
            return TextPart(text=_f.text)

        if _f.inline_data:
            data = _f.inline_data.data or self._raise_invalid_inline_data(field="data")
            mime_type = _f.inline_data.mime_type or self._raise_invalid_inline_data(
                field="mime_type"
            )
            return FilePart(data=data, mime_type=mime_type)

        if _f.function_call:
            return ToolExecutionSuggestion(
                id=_f.function_call.id or str(uuid.uuid4()),
                tool_name=_f.function_call.name
                or self._raise_invalid_function_call(field="name"),
                args=_f.function_call.args
                or self._raise_invalid_function_call(field="args"),
            )

        raise ValueError(
            f"The provided part: {_f} is not supported by the framework yet."
        )

    def _raise_invalid_inline_data(self, field: str) -> Never:
        """
        Raise an error for invalid inline data.

        This internal helper method raises a descriptive ValueError when
        required fields are missing from inline data.

        Args:
            field: The name of the missing field.

        Raises:
            ValueError: Always raises this error with a descriptive message.

        Returns:
            Never: This function never returns, it always raises an exception.
        """
        raise ValueError(f"Provided field '{field}' is None.")

    def _raise_invalid_function_call(self, field: str) -> Never:
        """
        Raise an error for invalid function call data.

        This internal helper method raises a descriptive ValueError when
        required fields are missing from a function call.

        Args:
            field: The name of the missing field.

        Raises:
            ValueError: Always raises this error with a descriptive message.

        Returns:
            Never: This function never returns, it always raises an exception.
        """
        raise ValueError(f"Provided field '{field}' is None.")
