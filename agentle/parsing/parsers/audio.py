import asyncio
import logging
import os
from pathlib import Path
from typing import Never

from rsb.functions.ext2mime import ext2mime
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.audio_description_agent_factory import (
    audio_description_agent_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent

logger = logging.getLogger(__name__)


@parses("flac", "mp3", "mpeg", "mpga", "m4a", "ogg", "wav", "webm")
class AudioFileParser(DocumentParser):
    audio_description_agent: Agent[AudioDescription] = Field(
        default_factory=audio_description_agent_factory,
    )

    async def parse_async(self, document_path: str) -> ParsedDocument:
        path = Path(document_path)
        file_contents: bytes = path.read_bytes()
        file_extension = path.suffix

        if file_extension in {
            "flac",
            "mpeg",
            "mpga",
            "m4a",
            "ogg",
            "wav",
            "webm",
        }:
            import aiofiles.os as aios
            from aiofiles import open as aio_open

            self._check_ffmpeg_installed()

            # Generate unique temporary filenames
            input_temp = os.path.join(
                tempfile.gettempdir(),
                f"input_{os.urandom(8).hex()}.{file_extension}",
            )
            output_temp = os.path.join(
                tempfile.gettempdir(), f"output_{os.urandom(8).hex()}.mp3"
            )

            # Write input file asynchronously
            async with aio_open(input_temp, "wb") as f:
                await f.write(file_contents)

            # Build FFmpeg command
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",  # Suppress unnecessary logs
                "-y",  # Overwrite output file if exists
                "-i",
                input_temp,
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",  # Quality preset (0-9, 0=best)
                output_temp,
            ]

            # Execute FFmpeg
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            # Handle conversion errors
            if process.returncode != 0:
                await aios.remove(input_temp)
                if await aios.path.exists(output_temp):
                    await aios.remove(output_temp)
                raise RuntimeError(
                    f"Audio conversion failed: {stderr.decode().strip()}"
                )

            # Read converted file
            async with aio_open(output_temp, "rb") as f:
                file_contents = await f.read()

            # Cleanup temporary files
            await aios.remove(input_temp)
            await aios.remove(output_temp)

        transcription = self.audio_description_agent.run(
            FilePart(data=file_contents, mime_type=ext2mime(file_extension))
        )

        return ParsedDocument(
            name=path.name,
            sections=[
                SectionContent(
                    number=1,
                    text=transcription.parsed.overall_description,
                    md=transcription.parsed.md,
                    images=[],
                )
            ],
        )

    def _could_not_transcript(self) -> Never:
        raise ValueError("Could not transcribe the audio")

    def _check_ffmpeg_installed(self) -> None:
        import subprocess

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.exception("FFmpeg is not installed or not in PATH.")
            if result.returncode != 0:
                raise RuntimeError()
        except FileNotFoundError:
            logger.exception("FFmpeg is not installed or not in PATH.")
            raise RuntimeError()
