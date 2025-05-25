from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentle.agents.agent import Agent
    from agentle.generations.models.structured_outputs_store.audio_description import (
        AudioDescription,
    )
    from agentle.generations.providers.base.generation_provider import (
        GenerationProvider,
    )


def audio_description_agent_default_factory(
    provider: GenerationProvider | None = None,
) -> Agent[AudioDescription]:
    """
    Creates and returns an Agent specialized for processing audio content.

    This factory function instantiates an Agent configured with appropriate models,
    instructions, and generation providers for analyzing audio files. The agent can
    transcribe speech, identify sounds, and provide descriptions of audio content.

    The agent is configured with:
    - Model: gemini-2.0-flash
    - Provider: GoogleGenaiGenerationProvider
    - Response schema: AudioDescription

    Returns:
        Agent[AudioDescription]: An agent configured for audio processing tasks

    Example:
        ```python
        from agentle.parsing.factories.audio_description_agent_default_factory import audio_description_agent_default_factory

        # Create the audio agent
        audio_agent = audio_description_agent_default_factory()

        # Process an audio file
        from agentle.generations.models.message_parts.file import FilePart

        with open("audio_sample.mp3", "rb") as f:
            audio_bytes = f.read()

        result = audio_agent.run(
            FilePart(data=audio_bytes, mime_type="audio/mpeg")
        )

        # Access the structured description
        print(result.parsed.overall_description)
        ```
    """
    from agentle.agents.agent import Agent
    from agentle.generations.models.structured_outputs_store.audio_description import (
        AudioDescription,
    )
    from agentle.generations.providers.google.google_genai_generation_provider import (
        GoogleGenaiGenerationProvider,
    )

    agent = Agent(
        model="gemini-2.0-flash",
        instructions="You are a helpful assistant that deeply understands audio files.",
        generation_provider=provider or GoogleGenaiGenerationProvider(),
        response_schema=AudioDescription,
    )

    return agent
