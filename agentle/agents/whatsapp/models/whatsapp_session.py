from collections.abc import MutableMapping
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_flood_tracker import WhatsAppFloodTracker


class WhatsAppSession(BaseModel):
    """WhatsApp conversation session."""

    session_id: str
    phone_number: str
    contact: WhatsAppContact
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True
    context_data: MutableMapping[str, Any] = Field(default_factory=dict)
    agent_context_id: str | None = None

    # Flood protection tracking
    flood_tracker: WhatsAppFloodTracker | None = None

    def get_or_create_flood_tracker(self) -> WhatsAppFloodTracker:
        """Get or create flood tracker for this session."""
        if self.flood_tracker is None:
            self.flood_tracker = WhatsAppFloodTracker(phone_number=self.phone_number)
        return self.flood_tracker
