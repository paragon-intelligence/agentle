from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)


def audio_description_agent_factory() -> Agent[AudioDescription]:
    return Agent(
        model="gemini-2.0-flash",
        instructions="You are a helpful assistant that deeply understands audio files.",
        generation_provider=GoogleGenaiGenerationProvider(),
        response_schema=AudioDescription,
    )
