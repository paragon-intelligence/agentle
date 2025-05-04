from typing import Literal

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parsers.file_parser import FileParser


async def parse_async(
    document_path: str,
    strategy: Literal["low", "high"] = "high",
    visual_description_agent: Agent[VisualMediaDescription] | None = None,
    audio_description_agent: Agent[AudioDescription] | None = None,
) -> ParsedDocument:
    """The easiest way to parse a document."""
    if visual_description_agent is None and audio_description_agent is None:
        return await FileParser(
            strategy=strategy,
        ).parse_async(document_path)
    elif visual_description_agent is not None and audio_description_agent is None:
        return await FileParser(
            strategy=strategy,
            visual_description_agent=visual_description_agent,
        ).parse_async(document_path)
    elif visual_description_agent is None and audio_description_agent is not None:
        return await FileParser(
            strategy=strategy,
            audio_description_agent=audio_description_agent,
        ).parse_async(document_path)
    else:
        return await FileParser(
            strategy=strategy,
            visual_description_agent=visual_description_agent,
            audio_description_agent=audio_description_agent,
        ).parse_async(document_path)
