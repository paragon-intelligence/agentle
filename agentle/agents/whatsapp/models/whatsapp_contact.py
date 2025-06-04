from rsb.models.base_model import BaseModel


class WhatsAppContact(BaseModel):
    """WhatsApp contact information."""

    phone: str
    name: str | None = None
    push_name: str | None = None
    profile_picture_url: str | None = None
