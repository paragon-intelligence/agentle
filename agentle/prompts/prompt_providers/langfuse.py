from __future__ import annotations

from typing import TYPE_CHECKING, override

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider

if TYPE_CHECKING:
    from langfuse import Langfuse


class LangfusePromptProvider(PromptProvider):
    client: Langfuse

    def __init__(self, client: Langfuse) -> None:
        super().__init__()
        self.client = client

    @override
    def provide(self, prompt_id: str, cache_ttl_seconds: int = 0) -> Prompt:
        langfuse_prompt = self.client.get_prompt(
            prompt_id, cache_ttl_seconds=cache_ttl_seconds
        )
        return Prompt(content=langfuse_prompt.prompt)
