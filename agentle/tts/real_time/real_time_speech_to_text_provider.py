import abc
from collections.abc import AsyncIterator

from agentle.stt.real_time.definitions.audio_format import AudioFormat
from agentle.stt.real_time.definitions.language_code import LanguageCode
from agentle.tts.real_time.definitions.speech_config import SpeechConfig
from agentle.tts.real_time.definitions.tts_stream_chunk import TTSStreamChunk
from agentle.tts.real_time.definitions.voice_info import VoiceInfo


class RealtimeTextToSpeechProvider(abc.ABC):
    """Abstract base class for text-to-speech providers."""

    @abc.abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        config: SpeechConfig,
    ) -> AsyncIterator[TTSStreamChunk]:
        """
        Convert text to speech with streaming output.

        Args:
            text: Text to convert to speech
            config: Speech synthesis configuration

        Yields:
            Audio chunks as they're generated
        """
        pass

    @abc.abstractmethod
    async def get_available_voices(
        self,
        language: LanguageCode | None = None,
    ) -> list[VoiceInfo]:
        """
        Get list of available voices.

        Args:
            language: Optional language filter

        Returns:
            List of available voices
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
