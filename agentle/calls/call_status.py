from enum import StrEnum


class CallStatus(StrEnum):
    """Current status of a phone call"""

    QUEUED = "queued"
    SCHEDULED = "scheduled"
    CONNECTING = "connecting"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
