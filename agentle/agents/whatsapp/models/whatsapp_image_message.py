from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppImageMessage(WhatsAppMediaMessage):
    """Image message model."""

    type: WhatsAppMessageType = WhatsAppMessageType.IMAGE
