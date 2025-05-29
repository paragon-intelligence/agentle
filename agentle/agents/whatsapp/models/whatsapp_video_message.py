from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppVideoMessage(WhatsAppMediaMessage):
    """Video message model."""

    type: WhatsAppMessageType = WhatsAppMessageType.VIDEO
    duration: int | None = None
