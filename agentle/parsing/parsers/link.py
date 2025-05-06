import tempfile
import os
from typing import Literal, override
from pathlib import Path
from urllib.parse import urlparse
from rsb.models.field import Field
from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.audio_description_agent_default_factory import (
    audio_description_agent_default_factory,
)
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parsers.file_parser import FileParser
from agentle.parsing.section_content import SectionContent


class LinkParser(DocumentParser):
    """
    A parser for links.

    This parser handles both URLs and local file paths. For URLs, it uses Playwright
    to fetch the content and render any dynamic elements. For local files, it delegates
    to the appropriate FileParser.
    """

    type: Literal["link"] = "link"
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory
    )
    audio_description_agent: Agent[AudioDescription] = Field(
        default_factory=audio_description_agent_default_factory
    )
    parse_timeout: float = Field(default=30)

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Parse the link.

        Args:
            document_path (str): URL or local file path to parse

        Returns:
            ParsedDocument: A structured representation of the parsed content
        """
        # Determine if the document_path is a URL or a local file path
        parsed_url = urlparse(document_path)
        is_url = parsed_url.scheme in ["http", "https"]

        if is_url:
            # Handle URL
            try:
                # We need to import these modules here to avoid dependency issues
                import aiohttp

                # Check if the URL points to a downloadable file
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.head(document_path) as response:
                            content_type = response.headers.get("Content-Type", "")
                            content_disposition = response.headers.get(
                                "Content-Disposition", ""
                            )

                            is_downloadable = (
                                # Content types that suggest a file
                                any(
                                    ct in content_type.lower()
                                    for ct in [
                                        "application/",
                                        "audio/",
                                        "video/",
                                        "image/",
                                    ]
                                )
                                or
                                # Content-Disposition header suggesting a file download
                                "attachment" in content_disposition
                                or
                                # Common file extensions in URL path
                                any(
                                    ext in Path(parsed_url.path).suffix.lower()
                                    for ext in [
                                        ".pdf",
                                        ".doc",
                                        ".docx",
                                        ".xls",
                                        ".xlsx",
                                        ".ppt",
                                        ".pptx",
                                        ".zip",
                                        ".rar",
                                        ".txt",
                                        ".csv",
                                        ".json",
                                        ".xml",
                                        ".jpg",
                                        ".jpeg",
                                        ".png",
                                        ".gif",
                                        ".mp3",
                                        ".mp4",
                                        ".avi",
                                        ".mov",
                                    ]
                                )
                            )

                            if is_downloadable:
                                # Download the file
                                return await self._download_and_parse_file(
                                    document_path
                                )
                    except Exception:
                        # If HEAD request fails, try with a GET request for content-type
                        pass
            except ImportError:
                # If aiohttp is not available, proceed with Playwright
                pass

            # Handle as a web page using Playwright
            return await self._parse_webpage(document_path)

        # Handle local file
        file_parser = FileParser(
            visual_description_agent=self.visual_description_agent,
            audio_description_agent=self.audio_description_agent,
        )
        return await file_parser.parse_async(document_path)

    async def _download_and_parse_file(self, url: str) -> ParsedDocument:
        """Download a file from a URL and parse it using the appropriate FileParser."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(urlparse(url).path).suffix
        ) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Import aiohttp here to handle ImportError gracefully
            import aiohttp

            # Download the file
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)

            # Parse the downloaded file
            file_parser = FileParser(
                visual_description_agent=self.visual_description_agent,
                audio_description_agent=self.audio_description_agent,
                parse_timeout=self.parse_timeout,
            )
            parsed_document = await file_parser.parse_async(temp_path)

            # Update the name to reflect the original URL
            original_name = Path(urlparse(url).path).name
            if original_name:
                parsed_document = ParsedDocument(
                    name=original_name, sections=parsed_document.sections
                )

            return parsed_document
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _parse_webpage(self, url: str) -> ParsedDocument:
        """Parse a webpage using Playwright."""
        # Import playwright here to handle ImportError gracefully
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            # Launch a browser with default viewport size
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Go to the URL and wait for the network to be idle
                # This helps ensure dynamic content is loaded
                await page.goto(url, timeout=self.parse_timeout * 1000)

                # Give extra time for JavaScript-heavy pages to finish rendering
                await page.wait_for_timeout(2000)

                # Get the page title
                title = await page.title()

                # Get the page content
                content = await page.content()

                # Get the visible text
                text = await page.evaluate("""() => {
                    return document.body.innerText;
                }""")

                # Create a SectionContent with the parsed data
                section = SectionContent(number=1, text=text, md=f"# {title}\n\n{text}")

                # Create a temporary HTML file for the content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".html"
                ) as tmp_file:
                    tmp_file.write(content.encode("utf-8"))
                    html_path = tmp_file.name

                try:
                    # Parse the HTML file to get any embedded media
                    file_parser = FileParser(
                        visual_description_agent=self.visual_description_agent,
                        audio_description_agent=self.audio_description_agent,
                        parse_timeout=self.parse_timeout,
                    )
                    html_parsed = await file_parser.parse_async(html_path)

                    # Combine sections if the HTML parser extracted additional information
                    if len(html_parsed.sections) > 0:
                        for i, html_section in enumerate(html_parsed.sections):
                            # If this is the first section, merge it with our main section
                            if i == 0:
                                # Update images and items from the HTML parser
                                section = SectionContent(
                                    number=section.number,
                                    text=section.text,
                                    md=section.md,
                                    images=html_section.images,
                                    items=html_section.items,
                                )
                            # For additional sections, add them as-is
                            else:
                                section = section + html_section
                finally:
                    # Clean up the temporary HTML file
                    if os.path.exists(html_path):
                        os.unlink(html_path)

                # Return the parsed document
                return ParsedDocument(
                    name=title or Path(urlparse(url).path).name or "webpage",
                    sections=[section],
                )
            finally:
                # Close the browser
                await browser.close()
