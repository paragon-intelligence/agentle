"""
PDF Document Parser Module

This module provides functionality for parsing PDF documents into structured representations.
It can extract text content, process embedded images, and organize the document by pages.
"""

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
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.image import Image
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent


class PDFFileParser(DocumentParser):
    """
    Parser for processing PDF documents into structured representations.

    This parser extracts content from PDF files, including text and embedded images.
    Each page in the PDF is represented as a separate section in the resulting ParsedDocument.
    With the "high" strategy, embedded images are analyzed using a visual description agent
    to extract text via OCR and generate descriptions.

    **Attributes:**

    *   `strategy` (Literal["high", "low"]):
        The parsing strategy to use. Defaults to "high".
        - "high": Performs thorough parsing including OCR and image analysis
        - "low": Performs basic text extraction without analyzing images

        **Example:**
        ```python
        parser = PDFFileParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        An optional custom agent for visual media description. If provided and strategy
        is "high", this agent will be used to analyze images embedded in the PDF.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on diagram and chart analysis in technical documents",
            response_schema=VisualMediaDescription
        )

        parser = PDFFileParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of a PDF file:
    ```python
    from agentle.parsing.parsers.pdf import PDFFileParser

    # Create a parser with default settings
    parser = PDFFileParser()

    # Parse a PDF file
    parsed_doc = parser.parse("document.pdf")

    # Access the pages as sections
    for i, section in enumerate(parsed_doc.sections):
        print(f"Page {i+1} content:")
        print(section.text[:100] + "...")  # Print first 100 chars of each page
    ```

    Processing a PDF with focus on image analysis:
    ```python
    from agentle.parsing.parsers.pdf import PDFFileParser

    # Create a parser with high-detail strategy
    parser = PDFFileParser(strategy="high")

    # Parse a PDF with images
    report = parser.parse("annual_report.pdf")

    # Extract and process images
    for i, section in enumerate(report.sections):
        page_num = i + 1
        print(f"Page {page_num} has {len(section.images)} images")

        for j, image in enumerate(section.images):
            print(f"  Image {j+1}:")
            if image.ocr_text:
                print(f"    OCR text: {image.ocr_text}")
    ```
    """

    type: Literal["pdf"] = "pdf"
    strategy: Literal["high", "low"] = Field(default="high")
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
        Asynchronously parse a PDF document and convert it to a structured representation.

        This method reads a PDF file, extracts text content from each page, and processes
        any embedded images. With the "high" strategy, images are analyzed using the
        visual description agent to extract text and generate descriptions.

        Args:
            document_path (str): Path to the PDF file to be parsed

        Returns:
            ParsedDocument: A structured representation of the PDF where:
                - Each PDF page is a separate section
                - Text content is extracted from each page
                - Images are extracted and (optionally) analyzed

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.pdf import PDFFileParser

            async def process_pdf():
                parser = PDFFileParser(strategy="high")
                result = await parser.parse_async("whitepaper.pdf")

                # Get the total number of pages
                print(f"Document has {len(result.sections)} pages")

                # Extract text from the first page
                if result.sections:
                    first_page = result.sections[0]
                    print(f"First page text: {first_page.text[:200]}...")

                    # Count images on the first page
                    print(f"First page has {len(first_page.images)} images")

            asyncio.run(process_pdf())
            ```
        """
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
                            ocr_text = agent_response.parsed.ocr_text or ""
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
