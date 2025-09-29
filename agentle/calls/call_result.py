import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.agent_run_output import AgentRunOutput
from agentle.calls.call_analysis import CallAnalysis
from agentle.calls.call_end_reason import CallEndReason
from agentle.calls.call_metrics import CallMetrics
from agentle.calls.call_status import CallStatus
from agentle.calls.call_transcript import CallTranscript
from agentle.calls.call_type import CallType
from agentle.calls.customer import Customer
from agentle.calls.phone_number import PhoneNumber


class CallResult(BaseModel):
    """Result of a phone call execution"""

    # Call identification
    call_id: str = Field(description="Unique call identifier")
    agent_id: str = Field(description="Agent that made the call")

    # Call details
    call_type: CallType = Field(description="Type of call")
    status: CallStatus = Field(description="Current call status")
    phone_number: PhoneNumber = Field(description="Phone number used for calling")
    customer: Customer = Field(description="Call recipient")

    # Timing
    created_at: datetime.datetime = Field(description="When call was created")
    started_at: datetime.datetime | None = Field(
        default=None, description="When call actually started"
    )
    ended_at: datetime.datetime | None = Field(
        default=None, description="When call ended"
    )

    # Call outcome
    end_reason: CallEndReason | None = Field(default=None, description="Why call ended")
    was_successful: bool = Field(
        default=False, description="Whether call achieved its goal"
    )

    # Call content
    transcript: CallTranscript | None = Field(
        default=None, description="Complete call transcript"
    )
    analysis: CallAnalysis | None = Field(
        default=None, description="Post-call analysis"
    )

    # Performance
    metrics: CallMetrics | None = Field(
        default=None, description="Call performance metrics"
    )

    # Control and monitoring
    control_url: str | None = Field(
        default=None, description="URL for live call control"
    )
    listen_url: str | None = Field(
        default=None, description="URL for live audio streaming"
    )

    # Suspension support (for HITL workflows)
    is_suspended: bool = Field(
        default=False, description="Whether call is suspended pending approval"
    )
    suspension_reason: str | None = Field(
        default=None, description="Reason for suspension"
    )
    resumption_token: str | None = Field(
        default=None, description="Token to resume suspended call"
    )

    # Streaming support
    is_streaming: bool = Field(
        default=False, description="Whether this is a streaming response"
    )
    is_final_chunk: bool = Field(
        default=True, description="Whether this is the final streaming chunk"
    )

    # Agent execution context (reuse existing AgentRunOutput structure)
    agent_run_output: AgentRunOutput[Any] | None = Field(
        default=None, description="Underlying agent execution result"
    )

    # Error information
    error_message: str | None = Field(
        default=None, description="Error message if call failed"
    )
    error_code: str | None = Field(default=None, description="Structured error code")
