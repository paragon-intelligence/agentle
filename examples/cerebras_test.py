from agentle.generations.providers.cerebras.cerebras_generation_provider import (
    CerebrasGenerationProvider,
)
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient


tracing_client = LangfuseObservabilityClient()

provider = CerebrasGenerationProvider(tracing_client=tracing_client)

generation = provider.create_generation_by_prompt(
    prompt="Hello, world!",
)

print(generation.text)
