"""
GIF Animation Parser Module

This module provides functionality for parsing animated GIF files into structured
representations. It can extract frames from GIF animations, analyze their content
using visual description agents, and organize them as sequential sections.
"""

import io
from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal, override

from rsb.functions.bytes2mime import bytes2mime
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


class GifFileParser(DocumentParser):
    """
    Parser for processing animated GIF files.

    This parser extracts frames from GIF animations and processes them as separate images.
    For larger animations, the parser intelligently selects representative frames (up to 3)
    spread throughout the animation to provide a comprehensive view of the content.
    Each selected frame is analyzed using a visual description agent to extract content
    and text via OCR.

    **Attributes:**

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe the visual content of GIF frames.
        This agent is responsible for generating descriptions and extracting text
        via OCR from the selected frames.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on movement and sequential changes in animations",
            response_schema=VisualMediaDescription
        )

        parser = GifFileParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of a GIF file:
    ```python
    from agentle.parsing.parsers.gif import GifFileParser

    # Create a parser with default settings
    parser = GifFileParser()

    # Parse a GIF file
    parsed_gif = parser.parse("animation.gif")

    # Access the frame descriptions
    for i, section in enumerate(parsed_gif.sections):
        print(f"Frame {i+1} description:")
        print(section.text)
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse a GIF file
    result = parse("reaction.gif")

    # Count the number of frames extracted
    frame_count = len(result.sections)
    print(f"Extracted {frame_count} frames from the GIF")

    # Access any OCR text found in the frames
    for section in result.sections:
        for image in section.images:
            if image.ocr_text:
                print(f"Text found in frame: {image.ocr_text}")
    ```
    """

    type: Literal["gif"] = "gif"

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    @override
    async def parse_async(
        self,
        document_path: str,
    ) -> ParsedDocument:
        """
        Asynchronously parse a GIF file and generate a structured representation.

        This method extracts frames from a GIF animation, selects representative frames
        (up to 3 for longer animations), and processes each frame using a visual
        description agent. Each frame is converted to PNG format for compatibility
        with the visual description agent.

        Args:
            document_path (str): Path to the GIF file to be parsed

        Returns:
            ParsedDocument: A structured representation where:
                - Each selected frame is represented as a separate section
                - Frames contain image data and descriptions
                - OCR text is extracted if text is present in the frames

        Raises:
            ValueError: If the file is not a GIF file

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.gif import GifFileParser

            async def analyze_gif():
                parser = GifFileParser()
                result = await parser.parse_async("animation.gif")

                # Print information about the frames
                print(f"GIF contains {len(result.sections)} analyzed frames")

                # Print descriptions of each frame
                for i, section in enumerate(result.sections):
                    print(f"Frame {i+1}:")
                    print(section.text)

            asyncio.run(analyze_gif())
            ```

        Note:
            For GIFs with more than 3 frames, this method selects frames at approximately
            1/3, 2/3, and the end of the animation to represent the full animation content.
            For GIFs with 3 or fewer frames, all frames are used.
        """
        from PIL import Image as PILImage

        path = Path(document_path)

        # Safety check: only proceed if it's a .gif
        # or you can attempt detection based on file headers
        extension = path.suffix
        if extension not in {"gif"}:
            raise ValueError("AnimatedImageFileParser only supports .gif files.")

        # --- 1. Load all frames from the GIF ---
        frames: list[PILImage.Image] = []
        with PILImage.open(document_path) as gif_img:
            try:
                while True:
                    frames.append(gif_img.copy())
                    gif_img.seek(gif_img.tell() + 1)
            except EOFError:
                pass  # we've reached the end of the animation

        num_frames = len(frames)
        if num_frames == 0:
            # No frames => no content
            return ParsedDocument(name=path.name, sections=[])

        # --- 2. Pick up to 3 frames, splitting the GIF into 3 segments ---
        # If there are fewer than 3 frames, just use them all.
        # If more than 3, pick three frames spaced across the animation.

        if num_frames <= 3:
            selected_frames = frames
        else:
            # Example approach: pick near 1/3, 2/3, end
            idx1 = max(0, (num_frames // 3) - 1)
            idx2 = max(0, (2 * num_frames // 3) - 1)
            idx3 = num_frames - 1
            # Ensure distinct indexes
            unique_indexes = sorted(set([idx1, idx2, idx3]))
            selected_frames = [frames[i] for i in unique_indexes]

        # --- 3. Convert each selected frame to PNG and (optionally) describe it ---
        pages: MutableSequence[SectionContent] = []
        for i, frame in enumerate(selected_frames, start=1):
            # Convert frame to PNG in-memory
            png_buffer = io.BytesIO()
            # Convert to RGBA if needed
            if frame.mode not in ("RGB", "RGBA"):
                frame = frame.convert("RGBA")
            frame.save(png_buffer, format="PNG")
            png_bytes = png_buffer.getvalue()

            frame_image_ocr: str | None = None
            # If strategy is HIGH, pass the frame to the agent
            text_description = ""
            if self.visual_description_agent:
                agent_input = FilePart(
                    mime_type=bytes2mime(png_bytes),
                    data=png_bytes,
                )
                agent_response = await self.visual_description_agent.run_async(
                    agent_input
                )
                frame_image_ocr = agent_response.parsed.ocr_text
                text_description = agent_response.parsed.md

            # Create an Image object
            frame_image = Image(
                name=f"{path.name}-frame{i}.png",
                contents=png_bytes,
                ocr_text=frame_image_ocr,
            )
            # Each frame is its own "page" in the final doc
            page_content = SectionContent(
                number=i,
                text=text_description,
                md=text_description,
                images=[frame_image],
            )
            pages.append(page_content)

        # --- 4. Return the multi-page ParsedFile ---
        return ParsedDocument(
            name=path.name,
            sections=pages,
        )
