import abc
from collections.abc import AsyncIterator

from agentle.stt.real_time.definitions.audio_format import AudioFormat
from agentle.stt.real_time.definitions.stt_config import STTConfig
from agentle.stt.real_time.definitions.stt_stream_chunk import STTStreamChunk
from agentle.stt.real_time.definitions.transcription_result import TranscriptionResult
from agentle.stt.real_time.definitions.language_code import LanguageCode


class RealtimeSpeechToTextProvider(abc.ABC):
    @abc.abstractmethod
    async def transcribe(
        self,
        audio_stream: AsyncIterator[STTStreamChunk],
        config: STTConfig,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe streaming audio to text (real-time processing).

        Args:
            audio_stream: Stream of audio chunks
            config: Transcription configuration

        Yields:
            Partial and final transcription results
        """
        ...

    @abc.abstractmethod
    async def get_supported_languages(self) -> list[LanguageCode]:
        """Get list of supported language codes."""
        ...

    @abc.abstractmethod
    async def get_supported_formats(self) -> list[AudioFormat]:
        """Get list of supported audio formats."""
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and responsive."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources and close connections."""
        ...
