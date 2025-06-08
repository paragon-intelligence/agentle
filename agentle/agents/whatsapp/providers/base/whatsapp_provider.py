"""
Base interface for WhatsApp providers.
"""

from abc import abstractmethod
from typing import Protocol

from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)


class WhatsAppProvider(Protocol):
    """
    Abstract base class for WhatsApp API providers.

    This interface defines the contract that all WhatsApp providers must implement,
    enabling support for different WhatsApp APIs (Evolution API, WhatsApp Business API, etc.)
    """

    @abstractmethod
    def get_instance_identifier(self) -> str:
        """Get the instance identifier for the WhatsApp provider."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the WhatsApp provider connection."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the WhatsApp provider connection."""
        pass

    @abstractmethod
    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """
        Send a text message.

        Args:
            to: Recipient phone number (with country code)
            text: Message text
            quoted_message_id: Optional ID of message to quote/reply to

        Returns:
            The sent message
        """
        pass

    @abstractmethod
    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """
        Send a media message (image, document, audio, video).

        Args:
            to: Recipient phone number
            media_url: URL of the media file
            media_type: Type of media (image, document, audio, video)
            caption: Optional caption for the media
            filename: Optional filename for documents
            quoted_message_id: Optional ID of message to quote/reply to

        Returns:
            The sent message
        """
        pass

    @abstractmethod
    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """
        Send typing indicator to show the bot is processing.

        Args:
            to: Recipient phone number
            duration: Duration in seconds to show typing
        """
        pass

    @abstractmethod
    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read."""
        pass

    @abstractmethod
    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information for a phone number."""
        pass

    @abstractmethod
    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number."""
        pass

    @abstractmethod
    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data."""
        pass

    @abstractmethod
    async def validate_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """Process incoming webhook data from WhatsApp."""
        pass

    @abstractmethod
    async def download_media(self, media_id: str) -> DownloadedMedia:
        """Download media content by ID."""
        pass

    @abstractmethod
    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        pass

    @abstractmethod
    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        pass
