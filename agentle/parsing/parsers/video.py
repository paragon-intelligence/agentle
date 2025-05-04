from pathlib import Path
from rsb.functions.ext2mime import ext2mime
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_factory import (
    visual_description_agent_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("mp4")
class VideoFileParser(DocumentParser):
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    async def parse_async(self, document_path: str) -> ParsedDocument:
        path = Path(document_path)
        extension = path.suffix
        if extension != "mp4":
            raise ValueError("VideoFileParser only supports .mp4 files.")

        file_contents = path.read_bytes()
        visual_media_description = await self.visual_description_agent.run_async(
            FilePart(data=file_contents, mime_type=ext2mime(extension))
        )

        return ParsedDocument(
            name=path.name,
            sections=[
                SectionContent(
                    number=1,
                    text=visual_media_description.parsed.md,
                    md=visual_media_description.parsed.md,
                    images=[],
                )
            ],
        )
