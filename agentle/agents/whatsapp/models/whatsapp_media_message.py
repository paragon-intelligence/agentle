from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage


class WhatsAppMediaMessage(WhatsAppMessage):
    """Media message model."""

    media_url: str
    media_mime_type: str
    media_size: int | None = None
    caption: str | None = None
    filename: str | None = None
