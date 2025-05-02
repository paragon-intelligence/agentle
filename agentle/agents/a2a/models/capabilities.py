"""
A2A Capabilities Model

This module defines the Capabilities class, which represents the capabilities of an agent
in the A2A protocol. It specifies what features and communication modes an agent supports,
enabling clients to adapt their interaction accordingly.
"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Capabilities(BaseModel):
    """
    Represents the capabilities of an agent in the A2A protocol.

    This class specifies what features and communication modes an agent supports,
    enabling clients to adapt their interaction accordingly. It includes flags for
    streaming support, push notifications, and state transition history tracking.

    Attributes:
        streaming: Optional flag indicating if the agent supports server-sent events (SSE)
        pushNotifications: Optional flag indicating if the agent can notify updates to client
        stateTransitionHistory: Optional flag indicating if the agent exposes status change history

    Example:
        ```python
        from agentle.agents.a2a.models.capabilities import Capabilities

        # Create a basic capabilities object
        basic_capabilities = Capabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        )

        # Create a fully-featured capabilities object
        advanced_capabilities = Capabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        )

        # Check if an agent supports streaming
        if advanced_capabilities.streaming:
            print("This agent supports streaming responses")

        # Check if push notifications are available
        if advanced_capabilities.pushNotifications:
            print("This agent can send push notifications")
        ```
    """

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
