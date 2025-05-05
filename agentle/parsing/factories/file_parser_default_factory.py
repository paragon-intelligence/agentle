from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentle.parsing.parsers.file_parser import FileParser


def file_parser_default_factory() -> FileParser:
    from agentle.parsing.parsers.file_parser import FileParser
    from agentle.parsing.factories.visual_description_agent_default_factory import (
        visual_description_agent_default_factory,
    )
    from agentle.parsing.factories.audio_description_agent_default_factory import (
        audio_description_agent_default_factory,
    )

    return FileParser(
        visual_description_agent=visual_description_agent_default_factory(),
        audio_description_agent=audio_description_agent_default_factory(),
    )
