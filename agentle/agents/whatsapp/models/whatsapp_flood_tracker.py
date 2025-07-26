from datetime import datetime
from collections.abc import MutableSequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class MessageTimestamp(BaseModel):
    """Track individual message timestamps for rate limiting."""

    timestamp: datetime
    message_id: str


class WhatsAppFloodTracker(BaseModel):
    """Track message rates and flood protection state for a user."""

    phone_number: str
    message_timestamps: MutableSequence[MessageTimestamp] = Field(default_factory=list)
    is_banned: bool = False
    ban_expires_at: datetime | None = None
    total_messages_sent: int = 0

    # Processing state for batching
    is_processing: bool = False
    processing_started_at: datetime | None = None
    queued_messages: MutableSequence[str] = Field(
        default_factory=list, description="Messages queued while processing"
    )

    def add_message(self, message_id: str) -> None:
        """Record a new message timestamp."""
        self.message_timestamps.append(
            MessageTimestamp(timestamp=datetime.now(), message_id=message_id)
        )
        self.total_messages_sent += 1

    def clean_old_timestamps(self, cutoff_time: datetime) -> None:
        """Remove timestamps older than the cutoff time."""
        self.message_timestamps = [
            ts for ts in self.message_timestamps if ts.timestamp > cutoff_time
        ]

    def get_recent_message_count(self, seconds: int = 60) -> int:
        """Get count of messages in the last N seconds."""
        cutoff = datetime.now().timestamp() - seconds
        cutoff_time = datetime.fromtimestamp(cutoff)
        return sum(1 for ts in self.message_timestamps if ts.timestamp > cutoff_time)

    def check_ban_expired(self) -> bool:
        """Check if ban has expired and update state."""
        if self.is_banned and self.ban_expires_at:
            if datetime.now() > self.ban_expires_at:
                self.is_banned = False
                self.ban_expires_at = None
                return True
        return False
