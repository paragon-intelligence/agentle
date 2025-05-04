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
    """The easiest way to parse a document."""
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
