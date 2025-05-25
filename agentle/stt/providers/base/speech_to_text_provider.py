import abc
from pathlib import Path

from agentle.stt.models.audio_transcription import AudioTranscription
from agentle.stt.models.transcription_config import TranscriptionConfig
from rsb.coroutines.run_sync import run_sync


class SpeechToTextProvider(abc.ABC):
    def transcribe(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        return run_sync(self.transcribe_async, audio_file=audio_file, config=config)

    @abc.abstractmethod
    async def transcribe_async(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        pass
