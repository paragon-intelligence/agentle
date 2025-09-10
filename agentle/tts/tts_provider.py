from __future__ import annotations

import abc

from agentle.tts.speech import Speech


class TTSProvider(abc.ABC):
    async def generate_speech(self, text: str, **kwargs) -> Speech: ...
