import abc
from collections.abc import AsyncIterator


class RealtimeSpeechToTextProvider(abc.ABC):
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
        pass

    @abc.abstractmethod
    async def get_supported_languages(self) -> list[LanguageCode]:
        """Get list of supported language codes."""
        pass

    @abc.abstractmethod
    async def get_supported_formats(self) -> list[AudioFormat]:
        """Get list of supported audio formats."""
        pass

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and responsive."""
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources and close connections."""
        pass
