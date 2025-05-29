# agentle/agents/whatsapp/providers/evolution.py
"""
Evolution API implementation for WhatsApp.
"""

from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, override
from urllib.parse import urljoin

import aiohttp

from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message_status import WhatsAppMessageStatus
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)


class EvolutionAPIProvider(WhatsAppProvider):
    """
    Evolution API implementation for WhatsApp messaging.

    This provider implements the WhatsApp interface using Evolution API,
    which provides a REST API for WhatsApp Web.
    """

    def __init__(self, config: EvolutionAPIConfig):
        """
        Initialize Evolution API provider.

        Args:
            config: Evolution API configuration
        """
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._sessions: dict[str, WhatsAppSession] = {}

    @override
    def get_instance_identifier(self) -> str:
        """Get the instance identifier for the WhatsApp provider."""
        return self.config.instance_name

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            headers = {
                "apikey": self.config.api_key,
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
        return self._session

    def _build_url(self, endpoint: str) -> str:
        """Build full URL for API endpoint."""
        return urljoin(self.config.base_url, f"/message/{endpoint}")

    async def initialize(self) -> None:
        """Initialize the Evolution API connection."""
        # Check instance status
        url = urljoin(self.config.base_url, "/instance/fetchInstances")
        async with self.session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to connect to Evolution API: {response.status}"
                )

            instances = await response.json()
            instance_found = any(
                inst.get("instance", {}).get("instanceName")
                == self.config.instance_name
                for inst in instances
            )

            if not instance_found:
                raise RuntimeError(f"Instance '{self.config.instance_name}' not found")

    async def shutdown(self) -> None:
        """Shutdown the Evolution API connection."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """Send a text message via Evolution API."""
        payload: dict[str, Any] = {
            "number": self._normalize_phone(to),
            "textMessage": {"text": text},
        }

        if quoted_message_id:
            payload["quoted"] = {"key": {"id": quoted_message_id}}

        url = self._build_url(f"sendText/{self.config.instance_name}")
        async with self.session.post(url, json=payload) as response:
            if response.status != 201:
                error = await response.text()
                raise RuntimeError(f"Failed to send message: {error}")

            data = await response.json()

            return WhatsAppTextMessage(
                id=data["key"]["id"],
                from_number=data["key"]["remoteJid"],
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                text=text,
                quoted_message_id=quoted_message_id,
            )

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """Send a media message via Evolution API."""
        # Determine endpoint based on media type
        endpoint_map = {
            "image": "sendImage",
            "document": "sendDocument",
            "audio": "sendAudio",
            "video": "sendVideo",
        }

        endpoint = endpoint_map.get(media_type)
        if not endpoint:
            raise ValueError(f"Unsupported media type: {media_type}")

        payload: MutableMapping[str, Any] = {
            "number": self._normalize_phone(to),
            "mediaMessage": {"mediaUrl": media_url},
        }

        if caption:
            payload["mediaMessage"]["caption"] = caption

        if filename and media_type == "document":
            payload["mediaMessage"]["fileName"] = filename

        if quoted_message_id:
            payload["quoted"] = {"key": {"id": quoted_message_id}}

        url = self._build_url(f"{endpoint}/{self.config.instance_name}")
        async with self.session.post(url, json=payload) as response:
            if response.status != 201:
                error = await response.text()
                raise RuntimeError(f"Failed to send media: {error}")

            data = await response.json()

            # Create appropriate media message type
            message_class_map = {
                "image": WhatsAppImageMessage,
                "document": WhatsAppDocumentMessage,
                "audio": WhatsAppAudioMessage,
                "video": WhatsAppVideoMessage,
            }

            message_class = message_class_map[media_type]
            return message_class(
                id=data["key"]["id"],
                from_number=data["key"]["remoteJid"],
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url=media_url,
                media_mime_type=f"{media_type}/*",
                caption=caption,
                filename=filename,
                quoted_message_id=quoted_message_id,
            )

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via Evolution API."""
        payload = {
            "number": self._normalize_phone(to),
            "delay": duration * 1000,  # Evolution API expects milliseconds
        }

        url = self._build_url(f"sendTyping/{self.config.instance_name}")
        async with self.session.post(url, json=payload) as response:
            if response.status != 201:
                # Typing indicator failures are non-critical
                pass

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via Evolution API."""
        payload = {"readMessages": [{"id": message_id, "fromMe": False}]}

        url = self._build_url(f"markAsRead/{self.config.instance_name}")
        async with self.session.put(url, json=payload) as response:
            if response.status != 200:
                # Read receipt failures are non-critical
                pass

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information via Evolution API."""
        normalized_phone = self._normalize_phone(phone)

        url = self._build_url(f"fetchProfile/{self.config.instance_name}")
        payload = {"number": normalized_phone}

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                return None

            data = await response.json()

            return WhatsAppContact(
                phone=normalized_phone,
                name=data.get("name"),
                push_name=data.get("pushName"),
                profile_picture_url=data.get("profilePictureUrl"),
            )

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number."""
        normalized_phone = self._normalize_phone(phone)

        # Check if session exists in memory
        if normalized_phone in self._sessions:
            session = self._sessions[normalized_phone]
            session.last_activity = datetime.now()
            return session

        # Create new session
        contact = await self.get_contact_info(phone)
        if not contact:
            contact = WhatsAppContact(phone=normalized_phone)

        session = WhatsAppSession(
            session_id=f"{self.config.instance_name}_{normalized_phone}",
            phone_number=normalized_phone,
            contact=contact,
        )

        self._sessions[normalized_phone] = session
        return session

    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data."""
        self._sessions[session.phone_number] = session

    async def process_webhook(self, payload: dict[str, Any]) -> WhatsAppWebhookPayload:
        """Process incoming webhook data from Evolution API."""
        # Evolution API webhook structure
        event_type: str | None = payload.get("event")
        if event_type is None:
            raise ValueError("Event type is required")

        instance_name: str | None = payload.get("instance")
        if instance_name is None:
            raise ValueError("Instance name is required")

        if instance_name != self.config.instance_name:
            raise ValueError(f"Webhook for wrong instance: {instance_name}")

        return WhatsAppWebhookPayload(
            event_type=event_type,
            instance_id=instance_name,
            data=payload.get("data", {}),
            timestamp=datetime.now(),
        )

    async def download_media(self, media_id: str) -> bytes:
        """Download media content by ID."""
        url = self._build_url(f"getBase64FromMediaMessage/{self.config.instance_name}")
        payload = {"key": {"id": media_id}}

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to download media: {response.status}")

            data = await response.json()
            import base64

            return base64.b64decode(data["base64"])

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        return self.config.webhook_url or ""

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        webhook_config = {
            "webhook": {
                "url": url,
                "webhook_by_events": True,
                "events": [
                    "messages.upsert",
                    "messages.update",
                    "send.message",
                    "connection.update",
                ],
            }
        }

        api_url = urljoin(
            self.config.base_url, f"/webhook/set/{self.config.instance_name}"
        )

        async with self.session.put(api_url, json=webhook_config) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"Failed to set webhook: {error}")

        self.config.webhook_url = url

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to Evolution API format."""
        # Remove non-numeric characters
        phone = "".join(c for c in phone if c.isdigit())

        # Ensure country code is present (default to Brazil +55 if not)
        if not phone.startswith("55") and len(phone) <= 11:
            phone = "55" + phone

        # Add @s.whatsapp.net suffix
        if not phone.endswith("@s.whatsapp.net"):
            phone = phone + "@s.whatsapp.net"

        return phone
