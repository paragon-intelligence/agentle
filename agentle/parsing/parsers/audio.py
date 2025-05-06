"""
Audio File Parser Module

This module provides functionality for parsing various audio file formats into structured
representations. It can transcribe speech, analyze audio content, and convert between formats
to ensure compatibility with the underlying AI models.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Never

from rsb.functions.ext2mime import ext2mime
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.audio_description_agent_default_factory import (
    audio_description_agent_default_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent

logger = logging.getLogger(__name__)


class AudioFileParser(DocumentParser):
    """
    Parser for processing various audio file formats.

    This parser can handle multiple audio formats including FLAC, MP3, WAV, OGG, and others.
    It uses an audio description agent to analyze and transcribe the audio content, and
    it has the ability to convert between formats using FFmpeg when necessary to ensure
    compatibility with the underlying AI models.

    **Attributes:**

    *   `audio_description_agent` (Agent[AudioDescription]):
        The agent used to analyze and transcribe the audio content. This agent is
        responsible for converting audio to text and generating descriptions.
        Defaults to the agent created by `audio_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.audio_description import AudioDescription

        custom_agent = Agent(
            model="gemini-2.0-flash",
            instructions="Transcribe audio with focus on technical terminology",
            response_schema=AudioDescription
        )

        parser = AudioFileParser(audio_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of an audio file:
    ```python
    from agentle.parsing.parsers.audio import AudioFileParser

    # Create a parser with default settings
    parser = AudioFileParser()

    # Parse an audio file
    parsed_audio = parser.parse("interview.mp3")

    # Access the transcription
    print(parsed_audio.sections[0].text)
    ```

    Working with different audio formats:
    ```python
    from agentle.parsing.parsers.audio import AudioFileParser
    from agentle.parsing.parse import parse

    # Using the generic parse function (easier)
    wav_result = parse("recording.wav")
    flac_result = parse("music.flac")
    ogg_result = parse("podcast.ogg")

    # All results have the same structure regardless of original format
    for result in [wav_result, flac_result, ogg_result]:
        print(f"Audio file: {result.name}")
        print(f"Transcription: {result.sections[0].text[:100]}...")
    ```

    **Requirements:**

    The AudioFileParser requires FFmpeg to be installed on the system for handling
    various audio formats. If FFmpeg is not installed, an error will be raised when
    trying to process certain audio formats that need conversion.
    """

    type: Literal["audio"] = "audio"

    audio_description_agent: Agent[AudioDescription] = Field(
        default_factory=audio_description_agent_default_factory,
    )

    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse an audio file and generate a structured representation.

        This method reads the audio file, processes it (possibly converting to a compatible format),
        and then uses the audio description agent to transcribe and analyze the content.

        Args:
            document_path (str): Path to the audio file to be parsed

        Returns:
            ParsedDocument: A structured representation containing the transcription and
                analysis of the audio content in a single section

        Raises:
            RuntimeError: If FFmpeg is required but not installed
            ValueError: If the audio file cannot be transcribed

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.audio import AudioFileParser

            async def transcribe_audio():
                parser = AudioFileParser()
                result = await parser.parse_async("speech.mp3")

                # Print the transcription
                print(f"Transcription: {result.sections[0].text}")

            asyncio.run(transcribe_audio())
            ```

        Note:
            For certain audio formats (flac, mpeg, mpga, m4a, ogg, wav, webm),
            this method will attempt to convert them to MP3 format using FFmpeg
            before processing. This conversion is done to ensure compatibility
            with the audio description agent.
        """
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

            # Generate unique temporary filename for output
            output_temp = os.path.join(
                tempfile.gettempdir(), f"output_{os.urandom(8).hex()}.mp3"
            )

            # Build FFmpeg command using original file directly
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",  # Suppress unnecessary logs
                "-y",  # Overwrite output file if exists
                "-i",
                document_path,
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
                if await aios.path.exists(output_temp):
                    await aios.remove(output_temp)
                raise RuntimeError(
                    f"Audio conversion failed: {stderr.decode().strip()}"
                )

            # Read converted file
            async with aio_open(output_temp, "rb") as f:
                file_contents = await f.read()

            # Cleanup temporary file
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
        """
        Helper method to raise a standardized error when transcription fails.

        This method never returns but always raises a ValueError with a message
        indicating that the audio could not be transcribed.

        Raises:
            ValueError: Always raised with the message "Could not transcribe the audio"

        Note:
            This is a utility method used internally for error handling.
        """
        raise ValueError("Could not transcribe the audio")

    def _check_ffmpeg_installed(self) -> None:
        """
        Check if FFmpeg is installed and available on the system.

        This method attempts to run the 'ffmpeg -version' command to verify
        that FFmpeg is installed and accessible. If FFmpeg is not found or
        not working correctly, a RuntimeError is raised.

        Raises:
            RuntimeError: If FFmpeg is not installed or not accessible

        Note:
            This is a utility method used internally to validate system requirements
            before attempting audio conversion operations.
        """
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
