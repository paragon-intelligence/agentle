from collections.abc import Sequence

from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppTextMessage(WhatsAppMessage):
    """Text message model."""

    type: WhatsAppMessageType = WhatsAppMessageType.TEXT
    text: str
    mentions: Sequence[str] = Field(default_factory=list)
