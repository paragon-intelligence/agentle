from enum import StrEnum


class STTProvider(StrEnum):
    """Speech-to-Text providers"""

    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"
    OPENAI_WHISPER = "openai-whisper"
    GOOGLE = "google"
    AZURE = "azure"
