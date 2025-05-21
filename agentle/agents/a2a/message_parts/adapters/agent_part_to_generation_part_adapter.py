"""
A2A Message Part to Generation Message Part Adapter

This module provides an adapter for converting A2A message parts to Generation message parts.
The adapter ensures compatibility between the A2A protocol's message parts and the
generation system's message parts.
"""

import base64

from rsb.adapters.adapter import Adapter

from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.message_parts.file_part import FilePart
from agentle.agents.a2a.message_parts.data_part import DataPart

from agentle.generations.models.message_parts.text import TextPart as GenerationTextPart
from agentle.generations.models.message_parts.file import FilePart as GenerationFilePart


class AgentPartToGenerationPartAdapter(
    Adapter[
        TextPart | FilePart | DataPart,
        GenerationTextPart | GenerationFilePart,
    ]
):
    """
    Adapter for converting A2A message parts to Generation message parts.

    This adapter transforms TextPart, FilePart, and DataPart objects from the A2A protocol
    into TextPart and FilePart objects used in the generation system.

    Example:
        ```python
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.message_parts.adapters.agent_part_to_generation_part_adapter import AgentPartToGenerationPartAdapter

        # Create an A2A message part
        a2a_part = TextPart(text="Hello, world!")

        # Convert to Generation message part
        adapter = AgentPartToGenerationPartAdapter()
        gen_part = adapter.adapt(a2a_part)

        print(gen_part.text)  # "Hello, world!"
        ```
    """

    def adapt(
        self,
        _f: TextPart | FilePart | DataPart,
    ) -> GenerationTextPart | GenerationFilePart:
        """
        Adapts an A2A message part to a Generation message part.

        This method converts A2A message parts (TextPart, FilePart, DataPart) into
        Generation message parts (TextPart, FilePart).

        Args:
            _f: The A2A message part to adapt

        Returns:
            The adapted Generation message part

        Example:
            ```python
            adapter = AgentPartToGenerationPartAdapter()
            gen_part = adapter.adapt(a2a_part)
            ```
        """
        match _f:
            case TextPart():
                return GenerationTextPart(
                    text=_f.text if isinstance(_f.text, str) else _f.text.text
                )
            case DataPart():
                # Treat DataPart as TextPart for now
                return GenerationTextPart(text=str(_f.data))
            case FilePart():
                # Get MIME type from the file
                mime_type = _f.file.mimeType

                if mime_type is None:
                    raise ValueError(
                        f"MIME type is required, but got None for file: {_f.file.name}"
                    )

                # Handle base64-encoded bytes
                if _f.file.bytes:
                    try:
                        decoded_bytes = base64.b64decode(_f.file.bytes)
                        return GenerationFilePart(
                            data=decoded_bytes, mime_type=mime_type
                        )
                    except Exception:
                        # If decoding fails, fall back to text
                        return GenerationTextPart(text=f"[File: {_f.file.name}]")

                # Handle URI
                elif _f.file.uri:
                    # Return a text representation for now
                    return GenerationTextPart(text=f"[File URI: {_f.file.uri}]")

                # Fallback
                return GenerationTextPart(text="[File content unavailable]")
