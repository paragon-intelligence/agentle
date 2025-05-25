from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal

import httpx
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict

from agentle.stt.models.audio_transcription import AudioTranscription
from agentle.stt.models.sentence_segment import SentenceSegment
from agentle.stt.models.transcription_config import TranscriptionConfig
from agentle.stt.providers.base.speech_to_text_provider import SpeechToTextProvider


class GroqSpeechToTextProvider(BaseModel, SpeechToTextProvider):
    api_key: str | None = None
    http_client: httpx.Client | None = None
    timeout: float | httpx.Timeout | None = None
    model: (
        Literal[
            "whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"
        ]
        | str
    ) = "whisper-large-v3"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def transcribe_async(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        from groq import Groq
        from groq.types.audio.transcription import Transcription

        _config = config or TranscriptionConfig()
        client = Groq(api_key=self.api_key, http_client=self.http_client)
        with open(audio_file, "rb") as file:
            transcription: Transcription = client.audio.transcriptions.create(
                file=file,
                model=self.model,
                prompt=_config.context_prompt,
                language=_config.language,
                response_format="verbose_json",
                temperature=_config.temperature,
            )

            # Extract segments and convert to SentenceSegment objects
            segments: MutableSequence[SentenceSegment] = []
            groq_segments = getattr(transcription, "segments", [])  # type: ignore
            for segment_dict in groq_segments:
                # segment_dict is a dictionary from the Groq API response
                sentence_segment = SentenceSegment(
                    id=int(segment_dict.get("id", 0)),  # type: ignore
                    sentence=str(segment_dict.get("text", "")).strip(),  # type: ignore
                    start=float(segment_dict.get("start", 0.0)),  # type: ignore
                    end=float(segment_dict.get("end", 0.0)),  # type: ignore
                    no_speech_prob=float(segment_dict.get("no_speech_prob", 0.0)),  # type: ignore
                )
                segments.append(sentence_segment)

            # Get duration from transcription or calculate from segments
            duration: float = float(getattr(transcription, "duration", 0.0))  # type: ignore
            if not duration and segments:
                duration = max(segment.end for segment in segments)

            # Create subtitles from segments
            def format_time(seconds: float) -> str:
                hours = int(seconds // 3600)
                remaining = seconds % 3600
                minutes = int(remaining // 60)
                remaining %= 60
                seconds_int = int(remaining)
                milliseconds = int((remaining - seconds_int) * 1000)
                return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

            srt_parts: MutableSequence[str] = []
            for idx, segment in enumerate(segments, start=1):
                start_time = format_time(segment.start)
                end_time = format_time(segment.end)
                part = f"{idx}\n{start_time} --> {end_time}\n{segment.sentence}"
                srt_parts.append(part)

            subtitles_content = "\n\n".join(srt_parts)

            from agentle.stt.models.subtitles import Subtitles

            subtitles = Subtitles(format="srt", subtitles=subtitles_content)

            # Estimate cost (Groq doesn't provide this, so we'll estimate based on duration)
            # Groq Whisper pricing is typically around $0.00025 per second
            estimated_cost: float = duration * 0.00025

            return AudioTranscription(
                text=transcription.text,
                segments=segments,
                cost=estimated_cost,
                duration=duration,
                subtitles=subtitles,
            )
