"""
Adapter module for converting Agentle Part objects to Google AI Part format.

This module provides the PartToGooglePartAdapter class, which transforms Agentle's
internal message part representations into the Part format expected by Google's
Generative AI APIs. This conversion is necessary when sending messages to Google's
AI models.

The adapter handles various types of content including text parts, file/image parts,
tool execution suggestions, and tool references. It ensures that Agentle's standardized
parts are correctly formatted according to Google's API requirements.

This adapter is typically used internally by the GoogleGenerationProvider when
preparing message content to be sent to Google's API.

Example:
```python
from agentle.generations.providers.google._adapters.part_to_google_part_adapter import (
    PartToGooglePartAdapter
)
from agentle.generations.models.message_parts.text import TextPart

# Create an Agentle text part
text_part = TextPart(text="Hello, world!")

# Create an adapter
adapter = PartToGooglePartAdapter()

# Convert to Google's format
google_part = adapter.adapt(text_part)

# Now use with Google's API
response = model.generate_content([google_part])
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai.types import Part as GooglePart


class PartToGooglePartAdapter(
    Adapter[TextPart | FilePart | ToolExecutionSuggestion | Tool, "GooglePart"]
):
    """
    Adapter for converting Agentle Part objects to Google AI Part format.

    This adapter transforms Agentle's internal message part representations
    (TextPart, FilePart, ToolExecutionSuggestion, Tool) into the Part format
    expected by Google's Generative AI APIs. It ensures that messages being
    sent to Google's models are properly formatted according to their API
    requirements.

    The adapter is a key component of Agentle's provider abstraction layer,
    allowing the framework to use a unified internal representation while still
    being able to communicate with Google's specific API formats.

    Supported conversions:
    - TextPart → GooglePart with text field
    - FilePart → GooglePart with inline_data field (Blob)
    - ToolExecutionSuggestion → GooglePart with function_call field
    - Tool → GooglePart with text field containing a tool reference

    Example:
        ```python
        # Convert a text part
        text_part = TextPart(text="What's the weather like?")
        adapter = PartToGooglePartAdapter()
        google_text_part = adapter.adapt(text_part)

        # Convert a file part (e.g., an image)
        from agentle.generations.models.message_parts.file import FilePart

        image_part = FilePart(
            data=b"<binary image data>",
            mime_type="image/jpeg"
        )
        google_image_part = adapter.adapt(image_part)

        # Convert a tool execution suggestion
        tool_suggestion = ToolExecutionSuggestion(
            id="call-123",
            tool_name="get_weather",
            args={"location": "London"}
        )
        google_function_call = adapter.adapt(tool_suggestion)
        ```
    """

    def adapt(
        self, _f: TextPart | FilePart | ToolExecutionSuggestion | Tool
    ) -> "GooglePart":
        """
        Convert an Agentle Part object to a Google AI Part.

        This method examines the type of the input part and converts it to the
        appropriate Google Part format. It handles TextPart, FilePart,
        ToolExecutionSuggestion, and Tool objects.

        Args:
            _f: The Agentle Part object to convert. This should be one of:
                TextPart, FilePart, ToolExecutionSuggestion, or Tool.

        Returns:
            GooglePart: A Google AI Part object containing the converted content
                in the format expected by Google's API.

        Example:
            ```python
            # Converting text
            text_part = TextPart(text="Hello!")
            google_part = adapter.adapt(text_part)
            # Result: GooglePart with text="Hello!"

            # Converting an image
            image_part = FilePart(data=image_bytes, mime_type="image/png")
            google_part = adapter.adapt(image_part)
            # Result: GooglePart with inline_data containing the image

            # Converting a tool execution suggestion
            tool_exec = ToolExecutionSuggestion(
                id="call-1",
                tool_name="calculator",
                args={"expression": "5+7"}
            )
            google_part = adapter.adapt(tool_exec)
            # Result: GooglePart with function_call field
            ```
        """
        from google.genai.types import Blob, FunctionCall
        from google.genai.types import Part as GooglePart

        match _f:
            case TextPart():
                return GooglePart(text=str(_f))
            case FilePart():
                return GooglePart(
                    inline_data=Blob(data=_f.data, mime_type=_f.mime_type)
                )
            case ToolExecutionSuggestion():
                return GooglePart(
                    function_call=FunctionCall(
                        id=_f.id,
                        name=_f.tool_name,
                        args=_f.args,
                    )
                )
            case Tool():
                return GooglePart(text=f"<tool>{_f.name}</tool>")
