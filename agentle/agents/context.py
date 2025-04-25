from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.step import Step
from agentle.generations.models.messages.message import Message


class Context(BaseModel):
    """
    Context is a collection of information that is used to guide the agent's behavior.
    """

    messages: Sequence[Message]
    steps: Sequence[Step] = Field(default_factory=list)
