from __future__ import annotations

import abc

from agentle.prompts.models.prompt import Prompt


class PromptProvider(abc.ABC):
    @abc.abstractmethod
    def provide(self, prompt_id: str, cache_ttl_seconds: int = 0) -> Prompt: ...
