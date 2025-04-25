from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Capabilities(BaseModel):
    streaming: bool | None = Field(default=None)
    """
    true if the agent supports SSE
    """
    pushNotifications: bool | None = Field(default=None)
    """
    true if the agent can notify updates to client
    """
    stateTransitionHistory: bool | None = Field(default=None)
    """
    true if the agent exposes status change history for tasks
    """
