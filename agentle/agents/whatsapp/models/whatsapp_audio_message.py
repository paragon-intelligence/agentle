from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppAudioMessage(WhatsAppMediaMessage):
    """Audio message model."""

    type: WhatsAppMessageType = WhatsAppMessageType.AUDIO
    duration: int | None = None
    is_voice_note: bool = False
