import inspect
from pathlib import Path
from typing import Any, Literal, MutableMapping, override

from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.audio_description_agent_factory import (
    audio_description_agent_factory,
)
from agentle.parsing.factories.visual_description_agent_factory import (
    visual_description_agent_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parser_registry


class FileParser(DocumentParser):
    strategy: Literal["low", "high"] = Field(default="high")
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    audio_description_agent: Agent[AudioDescription] = Field(
        default_factory=audio_description_agent_factory,
    )
    """
    The agent to use for generating the audio description of the document.
    Useful when you want to customize the prompt for the audio description.
    """

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
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
            "strategy": self.strategy,
            "visual_description_agent": self.visual_description_agent,
            "audio_description_agent": self.audio_description_agent,
        }

        kwargs: MutableMapping[str, Any] = {}
        for arg_name, arg_value in potential_args.items():
            if arg_name in valid_params:
                kwargs[arg_name] = arg_value

        return await parser_cls(**kwargs).parse_async(document_path)
