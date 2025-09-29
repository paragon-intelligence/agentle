from enum import StrEnum


class TTSProvider(StrEnum):
    """Text-to-Speech providers"""

    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    PLAYHT = "playht"
    AZURE = "azure"
    GOOGLE = "google"
