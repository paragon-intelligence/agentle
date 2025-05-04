from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)


def visual_description_agent_factory() -> Agent[VisualMediaDescription]: ...
