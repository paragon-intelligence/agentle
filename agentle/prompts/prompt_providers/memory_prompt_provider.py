from __future__ import annotations
from collections.abc import MutableMapping
from typing import override

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider


class MemoryPromptProvider(PromptProvider):
    prompt_mapping: MutableMapping[str, Prompt]

    def __init__(self, prompt_mapping: MutableMapping[str, str | Prompt]):
        new_mapping = {}
        for key, val in prompt_mapping.items():
            new_key = key

            # Converte valor string para Prompt
            if isinstance(val, str):
                val = Prompt(content=val)

            # Adiciona ao novo dicionário com a chave possivelmente modificada
            new_mapping[new_key] = val

        self.prompt_mapping = new_mapping

    @override
    def provide(self, prompt_id: str) -> Prompt:
        if prompt_id not in self.prompt_mapping.keys():
            raise KeyError(
                "The provided prompt is not present in the provided prompt mapping."
                + f"The provided prompts are {self.prompt_mapping.items()}"
            )

        prompt = self.prompt_mapping[prompt_id]
        return prompt
