"""
Static Image Parser Module

This module provides functionality for parsing static image files (PNG, JPEG, TIFF, BMP, etc.)
into structured representations. It can extract visual content, perform OCR to identify text,
and generate detailed descriptions of image content.
"""

import io
from pathlib import Path
from typing import Literal, override

from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from rsb.functions.ext2mime import ext2mime
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.image import Image
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent


class StaticImageParser(DocumentParser):
    """
    Parser for processing static image files in various formats.

    This parser handles multiple image formats including PNG, JPEG, TIFF, BMP, and others.
    It uses a visual description agent to analyze image content, extract text via OCR,
    and generate descriptive text about the image contents. For certain formats like TIFF,
    the parser automatically converts the image to a compatible format (PNG) before processing.

    **Attributes:**

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe the image content. This agent is responsible
        for generating descriptions and extracting text via OCR from the image.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Analyze images with focus on technical diagrams and charts",
            response_schema=VisualMediaDescription
        )

        parser = StaticImageParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of an image file:
    ```python
    from agentle.parsing.parsers.static_image import StaticImageParser

    # Create a parser with default settings
    parser = StaticImageParser()

    # Parse an image file
    parsed_image = parser.parse("photograph.jpg")

    # Access the description and OCR text
    print(f"Image description: {parsed_image.sections[0].text}")

    if parsed_image.sections[0].images[0].ocr_text:
        print(f"Text found in image: {parsed_image.sections[0].images[0].ocr_text}")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse different image formats
    png_result = parse("diagram.png")
    jpg_result = parse("photo.jpg")
    tiff_result = parse("scan.tiff")

    # All results have the same structure regardless of original format
    for result in [png_result, jpg_result, tiff_result]:
        print(f"Image file: {result.name}")
        print(f"Description: {result.sections[0].text[:100]}...")

        # Access the first (and only) image in the first section
        image = result.sections[0].images[0]
        if image.ocr_text:
            print(f"OCR text: {image.ocr_text}")
    ```
    """

    type: Literal["static_image"] = "static_image"

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse a static image file and generate a structured representation.

        This method reads an image file, converts it to a compatible format if necessary
        (e.g., TIFF to PNG), and processes it using a visual description agent to extract
        content and text via OCR.

        Args:
            document_path (str): Path to the image file to be parsed

        Returns:
            ParsedDocument: A structured representation where:
                - The image is contained in a single section
                - The section includes the image data and a description
                - OCR text is extracted if text is present in the image

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.static_image import StaticImageParser

            async def analyze_image():
                parser = StaticImageParser()
                result = await parser.parse_async("chart.png")

                # Access the image description
                print(f"Image description: {result.sections[0].text}")

                # Access OCR text if available
                image = result.sections[0].images[0]
                if image.ocr_text:
                    print(f"Text in image: {image.ocr_text}")

            asyncio.run(analyze_image())
            ```

        Note:
            For TIFF images, this method automatically converts them to PNG format
            before processing to ensure compatibility with the visual description agent.
        """
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
            current_mime_type = ext2mime(extension)
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
