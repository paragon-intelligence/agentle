import io
from pathlib import Path
from typing import override

from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from rsb.functions.ext_to_mime import ext_to_mime
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_factory import (
    visual_description_agent_factory,
)
from agentle.parsing.image import Image
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("png", "jpeg", "tiff", "bmp", "jpg", "jp2")
class StaticImageParser(DocumentParser):
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    multi_modal_provider: GenerationProvider = Field(
        default_factory=GoogleGenaiGenerationProvider,
    )
    """
    The multi-modal provider to use for generating the visual description of the document.
    Useful when you want us to customize the prompt for the visual description.
    """

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        from PIL import Image as PILImage

        path = Path(document_path)
        file_bytes = path.read_bytes()
        extension = path.suffix

        # Convert to PNG if TIFF
        if extension in {"tiff", "tif"}:
            # Use Pillow to open, then convert to PNG in memory
            with io.BytesIO(file_bytes) as input_buffer:
                with PILImage.open(input_buffer) as pil_img:
                    # Convert to RGBA or RGB if needed
                    if pil_img.mode not in ("RGB", "RGBA"):
                        pil_img = pil_img.convert("RGBA")

                    # Save as PNG into a new buffer
                    output_buffer = io.BytesIO()
                    pil_img.save(output_buffer, format="PNG")
                    converted_bytes = output_buffer.getvalue()

            # Use the converted PNG bytes
            image_bytes = converted_bytes
            current_mime_type = ext_to_mime(extension)
        else:
            # No conversion needed
            image_bytes = file_bytes

            # For demonstration, pick your MIME by extension
            if extension in {"png", "bmp"}:
                current_mime_type = "image/" + extension
            elif extension in {"jpg", "jpeg"}:
                current_mime_type = "image/jpeg"
            else:
                # Fallback to PNG or raise an error if you want
                current_mime_type = "image/png"

        # Create an Image object

        image_ocr: str | None = None
        # Generate a description if we have an agent + HIGH strategy
        text_content = ""
        agent_input = FilePart(
            mime_type=current_mime_type,
            data=image_bytes,
        )
        agent_response = await self.visual_description_agent.run_async(agent_input)
        description_md = agent_response.parsed.md
        image_ocr = agent_response.parsed.ocr_text
        text_content = description_md

        image_obj = Image(name=path.name, contents=image_bytes, ocr_text=image_ocr)
        # We treat it as a single "page" with one image
        page_content = SectionContent(
            number=1,
            text=text_content,
            md=text_content,
            images=[image_obj],
        )

        return ParsedDocument(
            name=path.name,
            sections=[page_content],
        )
