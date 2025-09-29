from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.call_config import CallConfig
from agentle.calls.call_result import CallResult
from agentle.calls.phone_number import PhoneNumber
from agentle.calls.realtime_conversation_provider import RealtimeConversationProvider


class PhoneCallState(BaseModel):
    """State that needs to be added to Agent for phone call support"""

    # Default phone configuration
    default_phone_number: PhoneNumber | None = Field(
        default=None, description="Default outbound number"
    )
    default_call_config: CallConfig = Field(
        default_factory=CallConfig, description="Default call configuration"
    )

    # Real-time conversation support
    conversation_provider: RealtimeConversationProvider | None = Field(
        default=None, description="Provider for real-time voice conversations"
    )

    # Active calls tracking
    active_calls: dict[str, CallResult] = Field(
        default_factory=dict, description="Currently active calls by call_id"
    )

    # Call history
    call_history: list[CallResult] = Field(
        default_factory=list, description="History of completed calls"
    )
