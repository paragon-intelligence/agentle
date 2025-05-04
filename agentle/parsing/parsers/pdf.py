import tempfile
from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal, override

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
from agentle.parsing.image import Image
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("pdf")
class PDFFileParser(DocumentParser):
    strategy: Literal["high", "low"] = Field(default="high")
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        import hashlib

        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{document_path}"
            with open(file_path, "wb") as f:
                f.write(Path(document_path).read_bytes())

            reader = PdfReader(file_path)
            section_contents: MutableSequence[SectionContent] = []
            image_cache: dict[str, tuple[str, str]] = {}

            for page_num, page in enumerate(reader.pages):
                page_images: MutableSequence[Image] = []
                image_descriptions: MutableSequence[str] = []

                if self.visual_description_agent and self.strategy == "high":
                    for image_num, image in enumerate(page.images):
                        image_bytes = image.data
                        image_hash = hashlib.sha256(image_bytes).hexdigest()

                        if image_hash in image_cache:
                            cached_md, cached_ocr = image_cache[image_hash]
                            image_md = cached_md
                            ocr_text = cached_ocr
                        else:
                            agent_input = FilePart(
                                mime_type=ext2mime(Path(image.name).suffix),
                                data=image.data,
                            )

                            agent_response = (
                                await self.visual_description_agent.run_async(
                                    agent_input
                                )
                            )

                            image_md = agent_response.parsed.md
                            ocr_text = agent_response.parsed.ocr_text
                            image_cache[image_hash] = (image_md, ocr_text or "")

                        image_descriptions.append(
                            f"Page Image {image_num + 1}: {image_md}"
                        )
                        page_images.append(
                            Image(
                                contents=image.data,
                                name=image.name,
                                ocr_text=ocr_text,
                            )
                        )

                page_text = [page.extract_text(), "".join(image_descriptions)]
                md = "".join(page_text)
                section_content = SectionContent(
                    number=page_num + 1,
                    text=md,
                    md=md,
                    images=page_images,
                )
                section_contents.append(section_content)

            return ParsedDocument(
                name=document_path,
                sections=section_contents,
            )
