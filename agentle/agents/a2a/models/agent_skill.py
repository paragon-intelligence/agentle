import uuid
from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.mimetype import MimeType


class AgentSkill(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """
    Unique identifier for the agent's skill
    """

    name: str
    """
    Human readable name of the skill
    """

    description: str
    """
    Description of the skill - will be used by the client or a human
    as a hint to understand what the skill does.
    """

    tags: Sequence[str]
    """
    Set of tagwords describing classes of capabilities for this specific skill
    """

    examples: Sequence[str] | None = Field(default=None)
    """
    Set of example scenarios that the skill can perform.
    """

    inputModes: Sequence[MimeType] | None = Field(default=None)
    """
    Set of interaction modes that the skill supports for input.
    """

    outputModes: Sequence[MimeType] | None = Field(default=None)
    """
    Set of interaction modes that the skill supports for output.
    """
