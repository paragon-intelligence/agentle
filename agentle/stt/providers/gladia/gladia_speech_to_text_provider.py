import asyncio
import os
import time
from pathlib import Path
from typing import override

import httpx
from rsb.functions.ext2mime import ext2mime
from rsb.models.base_model import BaseModel

from agentle.stt.models.audio_transcription import AudioTranscription
from agentle.stt.models.sentence_segment import SentenceSegment
from agentle.stt.models.subtitles import Subtitles
from agentle.stt.models.transcription_config import TranscriptionConfig
from agentle.stt.providers.base.speech_to_text_provider import SpeechToTextProvider


class GladiaSpeechToTextProvider(BaseModel, SpeechToTextProvider):
    api_key: str | None = None

    @override
    async def transcribe_async(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        api_key = self.api_key or os.getenv("GLADIA_API_KEY")
        _audio_file = str(audio_file)

        if not api_key:
            raise ValueError(
                "GLADIA_API_KEY is not set. Please provide a valid API key or set the environment variable."
            )

        # Base headers with API key
        api_headers = {
            "x-gladia-key": api_key,
        }

        # JSON headers for JSON requests
        json_headers = {
            **api_headers,
            "Content-Type": "application/json",
        }

        _config = config or TranscriptionConfig()

        async with httpx.AsyncClient(timeout=_config.timeout) as client:
            UPLOAD_URL = "https://api.gladia.io/v2/upload"

            extension = Path(audio_file).suffix
            mime_type = ext2mime(extension)

            file_bytes = Path(_audio_file).read_bytes()

            # For file upload, don't set Content-Type manually - httpx will set it correctly
            response = await client.post(
                UPLOAD_URL,
                headers=api_headers,  # Only use API key header, not Content-Type
                files={"audio": (_audio_file, file_bytes, mime_type)},
            )

            audio_upload_response = response.json()

            audio_url = audio_upload_response["audio_url"]

            # Initialize transcription proccess job asyncronously
            TRANSCRIPTION_URL = "https://api.gladia.io/v2/pre-recorded"
            response = await client.post(
                TRANSCRIPTION_URL,
                headers=json_headers,  # Use JSON headers for this request
                json={
                    "audio_url": audio_url,
                    "context_prompt": _config.context_prompt,
                    "subtitles": True,
                },
            )

            transcription_response = response.json()

            job_id = transcription_response["id"]

            RESULT_URL = f"https://api.gladia.io/v2/pre-recorded/{job_id}"

            # Poll for results with timeout
            start_time = time.time()
            max_time = start_time + _config.timeout

            while time.time() < max_time:
                response = await client.get(
                    RESULT_URL, headers=api_headers
                )  # Use API headers
                result = response.json()

                # Check if the job is complete
                status = result.get("status")

                if status == "done":
                    # Extract transcription from response
                    transcription_data = result.get("result", {}).get(
                        "transcription", {}
                    )

                    transcript = transcription_data.get("full_transcript", "")
                    utterances = transcription_data.get("utterances", [])

                    # Calculate duration from utterances if available
                    duration = 0.0
                    if utterances:
                        last_utterance = utterances[-1]
                        duration = last_utterance.get("end", 0.0)

                    # Create SentenceSegment objects from utterances
                    segments: list[SentenceSegment] = []
                    for i, utterance in enumerate(utterances):
                        start = utterance.get("start", 0.0)
                        end = utterance.get("end", 0.0)
                        text = utterance.get("text", "")

                        segment = SentenceSegment(
                            id=i,
                            sentence=text,
                            start=start,
                            end=end,
                            no_speech_prob=0.0,  # Gladia doesn't provide this, default to 0
                        )
                        segments.append(segment)

                    # Get subtitles if available
                    subtitle_data = ""
                    subtitles_list = transcription_data.get("subtitles", [])
                    if subtitles_list:
                        for subtitle in subtitles_list:
                            if subtitle.get("format") == "srt":
                                subtitle_data = subtitle.get("subtitles", "")
                                break

                    # Create and return the AudioTranscription object
                    return AudioTranscription(
                        text=transcript,
                        segments=segments,
                        cost=0.0,  # Gladia doesn't provide cost info in response
                        duration=duration,
                        subtitles=Subtitles(format="srt", subtitles=subtitle_data),
                    )

                elif status == "error":
                    error_message = result.get("error", {}).get(
                        "message", "Unknown error"
                    )
                    raise Exception(f"Transcription job failed: {error_message}")

                # If not complete, wait before checking again
                await asyncio.sleep(2)

            # If we've reached here, we've timed out
            raise TimeoutError(
                f"Transcription job timed out after {_config.timeout} seconds"
            )
