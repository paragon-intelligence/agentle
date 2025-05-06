"""
Microsoft Word Document Parser Module

This module provides functionality for parsing Microsoft Word documents (.doc, .docx) into
structured representations. It can extract text content, process embedded images, and
organize the document content.
"""

import hashlib
from pathlib import Path
from typing import Literal, override

from rsb.functions.ext2mime import ext2mime
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
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.image import Image
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent


class DocxFileParser(DocumentParser):
    """
    Parser for processing Microsoft Word documents (.doc, .docx).

    This parser extracts content from Word documents, including text and embedded images.
    With the "high" strategy, embedded images are analyzed using a visual description
    agent to extract text via OCR and generate descriptions. The parser represents the
    entire document as a single section containing all text and image content.

    **Attributes:**

    *   `strategy` (Literal["high", "low"]):
        The parsing strategy to use. Defaults to "high".
        - "high": Performs thorough parsing including OCR and image analysis
        - "low": Performs basic text extraction without analyzing images

        **Example:**
        ```python
        parser = DocxFileParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe the image content. If provided and
        strategy is "high", this agent will be used to analyze images embedded
        in the document.
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

        parser = DocxFileParser(visual_description_agent=custom_agent)
        ```

    *   `multi_modal_provider` (GenerationProvider):
        An alternative to using a visual_description_agent. This is a generation
        provider capable of handling multi-modal content (text and images).
        Defaults to GoogleGenaiGenerationProvider().

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

    **Usage Examples:**

    Basic parsing of a Word document:
    ```python
    from agentle.parsing.parsers.docx import DocxFileParser

    # Create a parser with default settings
    parser = DocxFileParser()

    # Parse a Word document
    parsed_doc = parser.parse("report.docx")

    # Access the text content
    print(parsed_doc.sections[0].text)

    # Access embedded images
    for image in parsed_doc.sections[0].images:
        print(f"Image: {image.name}")
        if image.ocr_text:
            print(f"  OCR text: {image.ocr_text}")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse a Word document
    result = parse("document.docx")

    # Access the document content
    print(f"Document: {result.name}")
    print(f"Text content: {result.sections[0].text[:100]}...")
    print(f"Contains {len(result.sections[0].images)} images")
    ```
    """

    type: Literal["docx"] = "docx"

    strategy: Literal["high", "low"] = Field(default="high")

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory,
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
    async def parse_async(
        self,
        document_path: str,
    ) -> ParsedDocument:
        """
        Asynchronously parse a Word document and generate a structured representation.

        This method reads a Word document, extracts its text content, and processes
        any embedded images. With the "high" strategy, images are analyzed using the
        visual description agent to extract text and generate descriptions.

        Args:
            document_path (str): Path to the Word document to be parsed

        Returns:
            ParsedDocument: A structured representation where:
                - The document is represented as a single section
                - The section includes all text content from the document
                - Embedded images are extracted and (optionally) analyzed

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.docx import DocxFileParser

            async def process_document():
                parser = DocxFileParser(strategy="high")
                result = await parser.parse_async("report.docx")

                # Access the document content
                print(f"Document text: {result.sections[0].text[:200]}...")

                # Process images
                print(f"Document contains {len(result.sections[0].images)} images")
                for img in result.sections[0].images:
                    if img.ocr_text:
                        print(f"Image text: {img.ocr_text}")

            asyncio.run(process_document())
            ```

        Note:
            This method uses the python-docx library to read Word documents. For optimal
            results, use .docx format rather than the older .doc format.
        """
        from docx import Document

        document = Document(document_path)
        image_cache: dict[str, tuple[str, str]] = {}  # (md, ocr_text)
        extension = Path(document_path).suffix

        paragraph_texts = [p.text for p in document.paragraphs if p.text.strip()]
        doc_text = "\n".join(paragraph_texts)

        doc_images: list[tuple[str, bytes]] = []
        for rel in document.part._rels.values():  # type: ignore[reportPrivateUsage]
            if "image" in rel.reltype:
                image_part = rel.target_part
                image_name = image_part.partname.split("/")[-1]
                image_bytes = image_part.blob
                doc_images.append((image_name, image_bytes))

        final_images: list[Image] = []
        image_descriptions: list[str] = []
        if self.visual_description_agent and self.strategy == "high":
            for idx, (image_name, image_bytes) in enumerate(doc_images, start=1):
                image_hash = hashlib.sha256(image_bytes).hexdigest()

                if image_hash in image_cache:
                    cached_md, cached_ocr = image_cache[image_hash]
                    image_md = cached_md
                    ocr_text = cached_ocr
                else:
                    agent_input = FilePart(
                        mime_type=ext2mime(extension),
                        data=image_bytes,
                    )
                    agent_response = await self.visual_description_agent.run_async(
                        agent_input
                    )
                    image_md = agent_response.parsed.md
                    ocr_text = agent_response.parsed.ocr_text or ""
                    image_cache[image_hash] = (image_md, ocr_text or "")

                image_descriptions.append(f"Docx Image {idx}: {image_md}")
                final_images.append(
                    Image(
                        name=image_name,
                        contents=image_bytes,
                        ocr_text=ocr_text,
                    )
                )

            if image_descriptions:
                doc_text += "\n\n" + "\n".join(image_descriptions)

        return ParsedDocument(
            name=document_path,
            sections=[
                SectionContent(
                    number=1,
                    text=doc_text,
                    md=doc_text,
                    images=final_images,
                )
            ],
        )
