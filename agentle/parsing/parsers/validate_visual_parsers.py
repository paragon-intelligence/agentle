from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider


def validate_visual_parsers(
    visual_description_agent: Agent[VisualMediaDescription],
    multi_modal_provider: GenerationProvider,
) -> None:
    if visual_description_agent and multi_modal_provider:
        raise ValueError(
            "Both visual_description_agent and multi_modal_provider cannot be passed at the same time"
        )
