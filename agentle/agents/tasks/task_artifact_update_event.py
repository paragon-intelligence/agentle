from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.artifact import Artifact


class TaskArtifactUpdateEvent(BaseModel):
    id: str = Field(...)
    """
    Task id
    """

    artifact: Artifact[Any]
    """
    artifact created by the agent
    """

    metadata: dict[str, Any]
    """
    extension metadata
    """
