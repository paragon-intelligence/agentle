"""
HTML Parser Module

This module provides functionality for parsing HTML files into structured
representations. It can extract text content, process embedded images, and
organize the document into a readable format.
"""

from pathlib import Path
from typing import Literal, override
from bs4 import BeautifulSoup, Tag
import hashlib
import base64
import re

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


class HTMLParser(DocumentParser):
    """
    Parser for processing HTML files.

    This parser extracts content from HTML files, converting HTML to markdown format
    for readability. With the "high" strategy, it also processes embedded images using
    a visual description agent to extract text via OCR and generate descriptions.

    **Attributes:**

    *   `strategy` (Literal["high", "low"]):
        The parsing strategy to use. Defaults to "high".
        - "high": Performs thorough parsing including OCR and image analysis
        - "low": Performs basic text extraction without analyzing images

        **Example:**
        ```python
        parser = HTMLParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe image content. If provided and
        strategy is "high", this agent will be used to analyze images embedded
        in the HTML document.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on technical content in web page images",
            response_schema=VisualMediaDescription
        )

        parser = HTMLParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of an HTML file:
    ```python
    from agentle.parsing.parsers.html import HTMLParser

    # Create a parser with default settings
    parser = HTMLParser()

    # Parse an HTML file
    parsed_html = parser.parse("webpage.html")

    # Access the content
    print(parsed_html.sections[0].text)  # Plain text content
    print(parsed_html.sections[0].md)    # Markdown-formatted content
    ```

    Processing a webpage with images:
    ```python
    from agentle.parsing.parsers.html import HTMLParser

    # Create a parser with high-detail strategy
    parser = HTMLParser(strategy="high")

    # Parse an HTML file with images
    result = parser.parse("article.html")

    # Extract and process images
    for image in result.sections[0].images:
        print(f"Image: {image.name}")
        if image.ocr_text:
            print(f"  OCR text: {image.ocr_text}")
    ```
    """

    type: Literal["html"] = "html"
    strategy: Literal["high", "low"] = Field(default="high")
    """if high, the parser will use the visual_description_agent to
    describe the images present in the html"""

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory
    )

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse an HTML file and generate a structured representation.

        This method reads an HTML file, converts it to markdown, extracts embedded images,
        and processes the images with a visual description agent if the "high" strategy is used.

        Args:
            document_path (str): Path to the HTML file to be parsed

        Returns:
            ParsedDocument: A structured representation containing the extracted content,
                with images and their descriptions if the "high" strategy is used

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.html import HTMLParser

            async def process_html():
                parser = HTMLParser(strategy="high")
                result = await parser.parse_async("webpage.html")

                # Access the content
                print(f"Title: {result.name}")
                print(f"Content: {result.sections[0].md[:200]}...")

                # Process images
                print(f"Found {len(result.sections[0].images)} images")

            asyncio.run(process_html())
            ```
        """
        from markdownify import markdownify as md_converter

        # Read the HTML file
        path = Path(document_path)
        html_content = path.read_text(encoding="utf-8", errors="replace")

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Get page title
        title_tag = soup.title
        title: str = (
            title_tag.string if title_tag and title_tag.string else str(path.name)
        )

        # Extract images
        images: list[Image] = []
        image_descriptions: list[str] = []
        image_cache: dict[
            str, tuple[str, str]
        ] = {}  # To avoid processing duplicate images

        if self.strategy == "high":
            img_tags = soup.find_all("img")
            for idx, img_tag in enumerate(img_tags, start=1):
                # Ensure we're working with a Tag, not NavigableString
                if not isinstance(img_tag, Tag):
                    continue

                src = img_tag.get("src", "")
                alt = img_tag.get("alt", "")

                # Skip empty sources or data URIs that are too small
                if not src or (
                    isinstance(src, str) and src.startswith("data:") and len(src) < 100
                ):
                    continue

                image_data = None
                image_name = f"image_{idx}"

                # Handle data URIs (embedded base64 images)
                if isinstance(src, str) and src.startswith("data:"):
                    mime_type = (
                        src.split(";")[0].split(":")[1]
                        if ";" in src
                        else "image/unknown"
                    )
                    extension = mime_type.split("/")[-1]
                    if extension == "jpeg":
                        extension = "jpg"

                    # Extract base64 data
                    base64_data = src.split(",")[1] if "," in src else ""
                    try:
                        image_data = base64.b64decode(base64_data)
                        image_name = f"embedded_image_{idx}.{extension}"
                    except Exception:
                        # Skip invalid base64 data
                        continue
                # Handle relative and absolute URLs - we can't process these directly
                # but we'll keep track of them for documentation
                else:
                    # We can't fetch remote images, so just document their existence
                    image_descriptions.append(f"Image {idx}: {alt or str(src)}")
                    continue

                # Process image with visual description agent
                if image_data and self.visual_description_agent:
                    # Generate a hash to avoid processing duplicate images
                    image_hash = hashlib.sha256(image_data).hexdigest()

                    if image_hash in image_cache:
                        cached_md, cached_ocr = image_cache[image_hash]
                        image_md = cached_md
                        ocr_text = cached_ocr
                    else:
                        extension = image_name.split(".")[-1]
                        agent_input = FilePart(
                            mime_type=ext2mime(f".{extension}"),
                            data=image_data,
                        )
                        agent_response = await self.visual_description_agent.run_async(
                            agent_input
                        )
                        image_md = agent_response.parsed.md
                        ocr_text = agent_response.parsed.ocr_text or ""
                        image_cache[image_hash] = (image_md, ocr_text or "")

                    image_descriptions.append(f"Image {idx}: {image_md}")
                    images.append(
                        Image(
                            name=image_name,
                            contents=image_data,
                            ocr_text=ocr_text or None,
                        )
                    )

        # Clean up HTML for better markdown conversion
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Convert HTML to markdown - ensure it's a string
        html_str = str(soup)
        markdown_content = str(md_converter(html_str))  # type: ignore

        # Add image descriptions to the content if available
        if image_descriptions:
            markdown_content += "\n\n## Images\n\n" + "\n\n".join(image_descriptions)

        # Create plain text from markdown (simple approach)
        text_content = re.sub(r"[#*_~`]", "", markdown_content)

        # Create section content
        section = SectionContent(
            number=1,
            text=text_content,
            md=markdown_content,
            images=images,
        )

        return ParsedDocument(
            name=title,
            sections=[section],
        )
