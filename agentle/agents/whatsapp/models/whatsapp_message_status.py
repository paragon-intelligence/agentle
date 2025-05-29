from enum import Enum


class WhatsAppMessageStatus(str, Enum):
    """Status of WhatsApp messages."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
