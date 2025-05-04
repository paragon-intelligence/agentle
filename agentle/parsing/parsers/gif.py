import io
from collections.abc import MutableSequence
from pathlib import Path
from typing import override

from rsb.functions.bytes2mime import bytes2mime
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


@parses("gif")
class GifFileParser(DocumentParser):
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
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
