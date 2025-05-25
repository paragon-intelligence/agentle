from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentle.agents.agent import Agent
    from agentle.generations.models.structured_outputs_store.visual_media_description import (
        VisualMediaDescription,
    )
    from agentle.generations.providers.base.generation_provider import (
        GenerationProvider,
    )


def visual_description_agent_default_factory(
    provider: GenerationProvider | None = None,
) -> Agent[VisualMediaDescription]:
    """
    Creates and returns an Agent specialized for processing visual media content.

    This factory function instantiates an Agent configured with appropriate models,
    instructions, and generation providers for analyzing images and visual content.
    The agent can identify objects, extract text via OCR, describe scenes, and provide
    detailed analysis of visual elements.

    The agent is configured with:
    - Model: gemini-2.0-pro-vision
    - Provider: GoogleGenaiGenerationProvider
    - Response schema: VisualMediaDescription

    Returns:
        Agent[VisualMediaDescription]: An agent configured for visual media processing tasks

    Example:
        ```python
        from agentle.parsing.factories.visual_description_agent_default_factory import visual_description_agent_default_factory

        # Create the visual media agent
        visual_agent = visual_description_agent_default_factory()

        # Process an image file
        from agentle.generations.models.message_parts.file import FilePart

        with open("image.jpg", "rb") as f:
            image_bytes = f.read()

        result = visual_agent.run(
            FilePart(data=image_bytes, mime_type="image/jpeg")
        )

        # Access the structured description
        print(result.parsed.md)
        print(result.parsed.ocr_text)
        ```
    """
    from agentle.agents.agent import Agent
    from agentle.generations.models.structured_outputs_store.visual_media_description import (
        VisualMediaDescription,
    )
    from agentle.generations.providers.google.google_genai_generation_provider import (
        GoogleGenaiGenerationProvider,
    )

    agent = Agent(
        model="gemini-2.0-pro-vision",
        instructions="You are a helpful assistant that deeply understands visual media.",
        generation_provider=provider or GoogleGenaiGenerationProvider(),
        response_schema=VisualMediaDescription,
    )

    return agent
