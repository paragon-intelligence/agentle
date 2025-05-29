from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppLocationMessage(WhatsAppMessage):
    """Location message model."""

    type: WhatsAppMessageType = WhatsAppMessageType.LOCATION
    latitude: float
    longitude: float
    name: str | None = None
    address: str | None = None
