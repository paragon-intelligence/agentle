import uuid

from rsb.models.base_model import BaseModel


class BestAgentOutput(BaseModel):
    """Identifies the best agent for a given task."""

    agent_id: uuid.UUID
    """The ID of the agent that is the best for the given task."""
