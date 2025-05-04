"""
Validation Logic for Visual Parser Configuration

This module contains validation functions to ensure proper configuration of visual parsers.
It helps to prevent invalid combinations of agents and providers that could lead to
runtime errors or unexpected behavior.
"""

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider


def validate_visual_parsers(
    visual_description_agent: Agent[VisualMediaDescription],
    multi_modal_provider: GenerationProvider,
) -> None:
    """
    Validates the configuration of visual parsers to prevent incompatible combinations.

    This function ensures that a visual description agent and a multi-modal provider
    are not both provided simultaneously, as this would create ambiguity in which
    component should handle the visual processing.

    Args:
        visual_description_agent (Agent[VisualMediaDescription]):
            An agent configured for visual media description. This agent is typically
            used to analyze images and generate structured descriptions.

        multi_modal_provider (GenerationProvider):
            A generation provider that can handle multi-modal content (text and images).
            This is an alternative approach to using a dedicated visual description agent.

    Raises:
        ValueError: If both visual_description_agent and multi_modal_provider are provided
                    at the same time.

    Example:
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription
        from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider
        from agentle.parsing.parsers.validate_visual_parsers import validate_visual_parsers

        # This is valid - only using an agent
        visual_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Describe images with attention to detail",
            generation_provider=GoogleGenaiGenerationProvider(),
            response_schema=VisualMediaDescription,
        )
        validate_visual_parsers(visual_agent, None)  # No error

        # This is valid - only using a provider
        provider = GoogleGenaiGenerationProvider()
        validate_visual_parsers(None, provider)  # No error

        # This will raise an error - can't use both
        validate_visual_parsers(visual_agent, provider)  # ValueError
        ```

    Note:
        In most parser implementations, you should use either a visual_description_agent
        OR a multi_modal_provider, but not both. This validation helps enforce this
        design constraint.
    """
    if visual_description_agent and multi_modal_provider:
        raise ValueError(
            "Both visual_description_agent and multi_modal_provider cannot be passed at the same time"
        )
