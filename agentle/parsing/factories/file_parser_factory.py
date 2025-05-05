from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentle.parsing.parsers.file_parser import FileParser


def file_parser_factory() -> FileParser:
    from agentle.parsing.parsers.file_parser import FileParser
    from agentle.parsing.factories.visual_description_agent_factory import (
        visual_description_agent_factory,
    )
    from agentle.parsing.factories.audio_description_agent_factory import (
        audio_description_agent_factory,
    )

    return FileParser(
        visual_description_agent=visual_description_agent_factory(),
        audio_description_agent=audio_description_agent_factory(),
    )
