import inspect
from pathlib import Path
from typing import Any, Literal, MutableMapping

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parser_registry


async def parse_async(
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

    kwargs: MutableMapping[str, Any] = {"strategy": strategy}
    if visual_description_agent:
        kwargs["visual_description_agent"] = visual_description_agent
    if audio_description_agent:
        kwargs["audio_description_agent"] = audio_description_agent

    return await parser_cls(**kwargs).parse_async(document_path)
