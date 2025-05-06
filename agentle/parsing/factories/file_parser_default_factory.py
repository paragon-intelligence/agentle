from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentle.generations.providers.base.generation_provider import (
        GenerationProvider,
    )
    from agentle.parsing.parsers.file_parser import FileParser


def file_parser_default_factory(
    visual_description_provider: GenerationProvider | None = None,
    audio_description_provider: GenerationProvider | None = None,
    parse_timeout: float = 30,
) -> FileParser:
    from agentle.parsing.factories.audio_description_agent_default_factory import (
        audio_description_agent_default_factory,
    )
    from agentle.parsing.factories.visual_description_agent_default_factory import (
        visual_description_agent_default_factory,
    )
    from agentle.parsing.parsers.file_parser import FileParser

    return FileParser(
        visual_description_agent=visual_description_agent_default_factory(
            provider=visual_description_provider
        ),
        audio_description_agent=audio_description_agent_default_factory(
            provider=audio_description_provider
        ),
        parse_timeout=parse_timeout,
    )
