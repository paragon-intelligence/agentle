from typing import Literal, override

from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_factory import (
    visual_description_agent_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses


@parses("html")
class HTMLParser(DocumentParser):
    type: Literal["html"] = "html"
    strategy: Literal["high", "low"] = Field(default="high")
    """if high, the parser will use the visual_description_agent to
    describe the images present in the html"""

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory
    )

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        from markdownify import markdownify as md

        pass
