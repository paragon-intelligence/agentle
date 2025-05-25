from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from rsb.functions.ext2mime import ext2mime
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.stt.models.audio_transcription import AudioTranscription
from agentle.stt.models.sentence_segment import SentenceSegment
from agentle.stt.models.subtitles import Subtitles
from agentle.stt.models.transcription_config import TranscriptionConfig
from agentle.stt.providers.base.speech_to_text_provider import SpeechToTextProvider


class _TranscriptionOutput(BaseModel):
    text: str = Field(description="The transcribed text.")
    segments: Sequence[SentenceSegment] = Field(
        description="The transcribed text broken down into segments."
    )
    subtitles: Subtitles = Field(description="The subtitles of the audio file.")


class GoogleSpeechToTextProvider(BaseModel, SpeechToTextProvider):
    model: str = "gemini-2.0-flash"
    use_vertex_ai: bool = False
    api_key: str | None = None
    project: str | None = None
    location: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _get_audio_duration(self, audio_file: str | Path) -> float:
        """Get the duration of an audio file using ffprobe."""
        try:
            # Use ffprobe to get audio duration
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(audio_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()

            if process.returncode != 0:
                # Fallback: try to estimate from file size (very rough estimate)
                file_path = Path(audio_file)
                if file_path.exists():
                    # Very rough estimate: assume 128kbps audio
                    file_size_bytes = file_path.stat().st_size
                    estimated_duration = file_size_bytes / (
                        128 * 1024 / 8
                    )  # 128kbps in bytes per second
                    return max(1.0, estimated_duration)  # At least 1 second
                return 1.0  # Default fallback

            # Parse ffprobe output
            probe_data = json.loads(stdout.decode())
            duration = float(probe_data["format"]["duration"])
            return duration

        except Exception:
            # Fallback: estimate from file size
            try:
                file_path = Path(audio_file)
                if file_path.exists():
                    file_size_bytes = file_path.stat().st_size
                    estimated_duration = file_size_bytes / (
                        128 * 1024 / 8
                    )  # 128kbps estimate
                    return max(1.0, estimated_duration)
            except Exception:
                pass
            return 1.0  # Final fallback

    async def transcribe_async(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        """Gemini has the amazing 'ability' to transcribe audio files."""
        generation_provider = GoogleGenerationProvider(
            use_vertex_ai=self.use_vertex_ai,
            api_key=self.api_key,
            project=self.project,
            location=self.location,
        )

        _config = config or TranscriptionConfig()
        language = _config.language or "en"

        path_audio_file = Path(audio_file)

        transcription = await generation_provider.create_generation_by_prompt_async(
            model=self.model,
            prompt=[
                TextPart(
                    text=f"""
            You are a helpful assistant that transcribes audio files.
            The audio file is {audio_file}.
            The language of the audio file is {language}.
            """
                ),
                FilePart(
                    data=path_audio_file.read_bytes(),
                    mime_type=ext2mime(path_audio_file.suffix),
                ),
            ],
            response_schema=_TranscriptionOutput,
        )

        prompt_tokens_used = transcription.usage.prompt_tokens
        completion_tokens_used = transcription.usage.completion_tokens

        ppmi = generation_provider.price_per_million_tokens_input(
            self.model, estimate_tokens=prompt_tokens_used
        )

        ppco = generation_provider.price_per_million_tokens_output(
            self.model, estimate_tokens=completion_tokens_used
        )

        cost: float = (
            prompt_tokens_used * ppmi + completion_tokens_used * ppco
        ) / 1_000_000

        transcription_output = transcription.parsed

        # Get audio duration
        duration = await self._get_audio_duration(audio_file)

        return AudioTranscription(
            text=transcription_output.text,
            segments=transcription_output.segments,
            subtitles=transcription_output.subtitles,
            cost=cost,
            duration=duration,
        )
