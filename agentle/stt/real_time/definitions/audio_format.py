from enum import StrEnum

class AudioFormat(StrEnum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"
    PCM_S16LE = "pcm_s16le"
    MULAW = "mulaw"
    ALAW = "alaw"
