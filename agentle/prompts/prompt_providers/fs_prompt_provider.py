from pathlib import Path

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider


class FSPromptProvider(PromptProvider):
    base_path: str | None

    def __init__(self, base_path: str | None = None) -> None:
        super().__init__()
        self.base_path = base_path

    def provide(self, prompt_id: str, cache_ttl_seconds: int = 0) -> Prompt:
        return Prompt(
            content=Path(
                f"{self.base_path}/{prompt_id.replace('.md', '')}.md"
            ).read_text()
        )
