from collections.abc import AsyncGenerator, AsyncIterable


from rsb.models.base_model import BaseModel
import abc

from agentle.stt.models.audio_transcription import AudioTranscription


class RealtimeSpeechToTextProvider(BaseModel, abc.ABC):
    @abc.abstractmethod
    async def transcribe_async(
        self, audio_stream: AsyncIterable[bytes], language: str = "en"
    ) -> AsyncGenerator[AudioTranscription, None]:
        """
        Transcribe audio in real-time from a streaming source.

        Yields incremental AudioTranscription objects as speech is detected and processed.
        """
        ...
