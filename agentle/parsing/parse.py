from pathlib import Path
from typing import Literal

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parsers.parse_async import parse_async
from agentle.parsing.parses import parser_registry
from rsb.coroutines.run_sync import run_sync


def parse(
    document_path: str,
    strategy: Literal["low", "high"] = "high",
    visual_description_agent: Agent[VisualMediaDescription] | None = None,
    audio_description_agent: Agent[AudioDescription] | None = None,
) -> ParsedDocument:
    """
    Parse any supported document type into a structured ParsedDocument representation.

    This function serves as the main entry point for document parsing in the Agentle framework.
    It automatically selects the appropriate parser based on the file extension and applies
    the requested parsing strategy.

    Args:
        document_path (str): Path to the document file to be parsed. The file extension
            is used to determine which parser to use.

        strategy (Literal["low", "high"], optional): Parsing strategy to use. Defaults to "high".
            - "high": More thorough parsing that may include OCR, image analysis,
                      and other CPU-intensive operations
            - "low": Faster parsing that skips some intensive operations

        visual_description_agent (Agent[VisualMediaDescription] | None, optional):
            Custom agent for analyzing visual content. If provided, this agent will be used
            instead of the default visual description agent. Useful for customizing
            the image analysis behavior. Defaults to None.

        audio_description_agent (Agent[AudioDescription] | None, optional):
            Custom agent for analyzing audio content. If provided, this agent will be used
            instead of the default audio description agent. Useful for customizing
            the audio analysis behavior. Defaults to None.

    Returns:
        ParsedDocument: A structured representation of the parsed document with:
            - sections: list of content sections
            - images: extracted images with optional OCR text
            - structured items: headings, tables, and text blocks

    Raises:
        ValueError: If the file extension is not supported by any registered parser

    Examples:
        Parse a PDF document with default settings:
        ```python
        from agentle.parsing.parse import parse

        parsed_doc = parse("document.pdf")
        print(f"Document name: {parsed_doc.name}")
        print(f"Number of sections: {len(parsed_doc.sections)}")
        ```

        Parse an image with a "low" strategy (faster processing):
        ```python
        parsed_image = parse("image.jpg", strategy="low")
        ```

        Parse with a custom visual description agent:
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription
        from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Describe images with focus on technical details",
            generation_provider=GoogleGenaiGenerationProvider(),
            response_schema=VisualMediaDescription,
        )

        parsed_image = parse("image.png", visual_description_agent=custom_agent)
        ```
    """
    path = Path(document_path)
    parser_cls = parser_registry.get(path.suffix)

    if not parser_cls:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    return run_sync(
        parse_async,
        document_path=document_path,
        strategy=strategy,
        visual_description_agent=visual_description_agent,
        audio_description_agent=audio_description_agent,
    )
