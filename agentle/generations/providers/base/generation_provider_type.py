from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentle.generations.providers.amazon.bedrock_generation_provider import (
        BedrockGenerationProvider,
    )
    from agentle.generations.providers.cerebras.cerebras_generation_provider import (
        CerebrasGenerationProvider,
    )
    from agentle.generations.providers.failover.failover_generation_provider import (
        FailoverGenerationProvider,
    )
    from agentle.generations.providers.google.google_generation_provider import (
        GoogleGenerationProvider,
    )
    from agentle.generations.providers.ollama.ollama_generation_provider import (
        OllamaGenerationProvider,
    )
    from agentle.generations.providers.openai.openai import OpenAIGenerationProvider

type GenerationProviderType = (
    BedrockGenerationProvider
    | CerebrasGenerationProvider
    | FailoverGenerationProvider
    | GoogleGenerationProvider
    | OllamaGenerationProvider
    | OpenAIGenerationProvider
)
