from collections.abc import MutableMapping
import inspect
from pathlib import Path
from typing import Any, Literal

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

    # Get the signature of the parser constructor
    parser_signature = inspect.signature(parser_cls.__init__)
    valid_params = parser_signature.parameters.keys()

    # Only include arguments that are accepted by the parser constructor
    potential_args = {
        "strategy": strategy,
        "visual_description_agent": visual_description_agent,
        "audio_description_agent": audio_description_agent,
    }

    kwargs: MutableMapping[str, Any] = {}
    for arg_name, arg_value in potential_args.items():
        if arg_name in valid_params and arg_value is not None:
            kwargs[arg_name] = arg_value

    return await parser_cls(**kwargs).parse_async(document_path)
