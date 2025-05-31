# agentle/agents/whatsapp/providers/evolution.py
"""
Evolution API implementation for WhatsApp.
"""

import logging
from collections.abc import Mapping, MutableMapping
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
from agentle.sessions.session_manager import SessionManager
from agentle.sessions.in_memory_session_store import InMemorySessionStore

logger = logging.getLogger(__name__)


class EvolutionAPIError(Exception):
    """Exception raised for Evolution API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class EvolutionAPIProvider(WhatsAppProvider):
    """
    Evolution API implementation for WhatsApp messaging.

    This provider implements the WhatsApp interface using Evolution API,
    which provides a REST API for WhatsApp Web.

    Features:
    - Automatic session management with configurable storage
    - Proper error handling and retry logic
    - Webhook verification
    - Message parsing and validation
    - Media handling
    """

    config: EvolutionAPIConfig
    session_manager: SessionManager[WhatsAppSession]
    session_ttl_seconds: int
    _session: aiohttp.ClientSession | None

    def __init__(
        self,
        config: EvolutionAPIConfig,
        session_manager: SessionManager[WhatsAppSession] | None = None,
        session_ttl_seconds: int = 3600,
    ):
        """
        Initialize Evolution API provider.

        Args:
            config: Evolution API configuration
            session_manager: Optional session manager (creates in-memory if not provided)
            session_ttl_seconds: Default TTL for sessions in seconds
        """
        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._session: aiohttp.ClientSession | None = None

        # Initialize session manager
        if session_manager is None:
            session_store = InMemorySessionStore[WhatsAppSession]()
            self.session_manager = SessionManager(
                session_store=session_store, default_ttl_seconds=session_ttl_seconds
            )
        else:
            self.session_manager = session_manager

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
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    def _build_url(self, endpoint: str, use_message_prefix: bool = True) -> str:
        """
        Build full URL for API endpoint.

        Args:
            endpoint: The API endpoint
            use_message_prefix: Whether to prefix with /message/ (default: True)
        """
        if use_message_prefix:
            return urljoin(self.config.base_url, f"/message/{endpoint}")
        else:
            return urljoin(self.config.base_url, f"/{endpoint}")

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> Mapping[str, Any]:
        """
        Make HTTP request with proper error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Full URL for the request
            data: Optional JSON data to send
            expected_status: Expected HTTP status code

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the request fails
        """
        try:
            match method.upper():
                case "GET":
                    async with self.session.get(url) as response:
                        return await self._handle_response(response, expected_status)
                case "POST":
                    async with self.session.post(url, json=data) as response:
                        return await self._handle_response(response, expected_status)
                case "PUT":
                    async with self.session.put(url, json=data) as response:
                        return await self._handle_response(response, expected_status)
                case "DELETE":
                    async with self.session.delete(url) as response:
                        return await self._handle_response(response, expected_status)
                case _:
                    raise ValueError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error for {method} {url}: {e}")
            raise EvolutionAPIError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise EvolutionAPIError(f"Unexpected error: {e}")

    async def _handle_response(
        self, response: aiohttp.ClientResponse, expected_status: int
    ) -> Mapping[str, Any]:
        """
        Handle HTTP response with proper error handling.

        Args:
            response: aiohttp response object
            expected_status: Expected HTTP status code

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the response indicates an error
        """
        if response.status == expected_status:
            try:
                return await response.json()
            except Exception:
                # If response is not JSON, return empty dict
                return {}

        # Handle error responses
        try:
            error_data = await response.json()
        except Exception:
            error_data = {"error": await response.text()}

        error_message = f"Evolution API error: {response.status}"
        if "error" in error_data:
            error_message += f" - {error_data['error']}"
        elif "message" in error_data:
            error_message += f" - {error_data['message']}"

        logger.error(f"Evolution API error: {error_message}")
        raise EvolutionAPIError(
            error_message, status_code=response.status, response_data=error_data
        )

    async def initialize(self) -> None:
        """Initialize the Evolution API connection."""
        try:
            # Check instance status
            url = self._build_url("instance/fetchInstances", use_message_prefix=False)
            response_data = await self._make_request("GET", url)

            # Look for our instance in the response
            instances = (
                response_data if isinstance(response_data, list) else [response_data]
            )
            instance_found = False

            for instance_data in instances:
                if isinstance(instance_data, dict):
                    instance_info = instance_data.get("instance", {})
                    if instance_info.get("instanceName") == self.config.instance_name:
                        instance_found = True
                        break

            if not instance_found:
                available_instances = []
                for inst in instances:
                    if isinstance(inst, dict):
                        inst_info = inst.get("instance", {})
                        inst_name = inst_info.get("instanceName")
                        if inst_name:
                            available_instances.append(inst_name)

                error_msg = (
                    f"Instance '{self.config.instance_name}' not found. "
                    f"Available instances: {available_instances}"
                )
                raise EvolutionAPIError(error_msg)

            logger.info(
                f"Evolution API provider initialized for instance: {self.config.instance_name}"
            )

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Evolution API provider: {e}")
            raise EvolutionAPIError(f"Initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Evolution API connection."""
        try:
            if self._session:
                await self._session.close()
                self._session = None

            # Close session manager
            await self.session_manager.close()

            logger.info("Evolution API provider shutdown complete")

        except Exception as e:
            logger.error(f"Error during Evolution API provider shutdown: {e}")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """Send a text message via Evolution API."""
        try:
            payload: Mapping[str, Any] = {
                "number": self._normalize_phone(to),
                "textMessage": {"text": text},
            }

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"sendText/{self.config.instance_name}")
            response_data = await self._make_request(
                "POST", url, payload, expected_status=201
            )

            message = WhatsAppTextMessage(
                id=response_data["key"]["id"],
                from_number=response_data["key"]["remoteJid"],
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                text=text,
                quoted_message_id=quoted_message_id,
            )

            logger.debug(f"Text message sent successfully: {message.id}")
            return message

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
            raise EvolutionAPIError(f"Failed to send text message: {e}")

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
        try:
            # Determine endpoint based on media type
            endpoint_map = {
                "image": "sendImage",
                "document": "sendDocument",
                "audio": "sendAudio",
                "video": "sendVideo",
            }

            endpoint = endpoint_map.get(media_type)
            if not endpoint:
                raise EvolutionAPIError(f"Unsupported media type: {media_type}")

            payload: MutableMapping[str, Any] = {
                "number": self._normalize_phone(to),
                "mediaMessage": {
                    "mediaurl": media_url
                },  # Note: Evolution API uses "mediaurl" not "mediaUrl"
            }

            if caption:
                payload["mediaMessage"]["caption"] = caption

            if filename and media_type == "document":
                payload["mediaMessage"]["fileName"] = filename

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"{endpoint}/{self.config.instance_name}")
            response_data = await self._make_request(
                "POST", url, payload, expected_status=201
            )

            # Create appropriate media message type
            message_class_map = {
                "image": WhatsAppImageMessage,
                "document": WhatsAppDocumentMessage,
                "audio": WhatsAppAudioMessage,
                "video": WhatsAppVideoMessage,
            }

            message_class = message_class_map[media_type]
            message = message_class(
                id=response_data["key"]["id"],
                from_number=response_data["key"]["remoteJid"],
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url=media_url,
                media_mime_type=f"{media_type}/*",
                caption=caption,
                filename=filename,
                quoted_message_id=quoted_message_id,
            )

            logger.debug(f"Media message sent successfully: {message.id}")
            return message

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to send media message: {e}")
            raise EvolutionAPIError(f"Failed to send media message: {e}")

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via Evolution API."""
        try:
            payload = {
                "number": self._normalize_phone(to),
                "delay": duration * 1000,  # Evolution API expects milliseconds
            }

            url = self._build_url(f"sendPresence/{self.config.instance_name}")
            await self._make_request("POST", url, payload, expected_status=201)

            logger.debug(f"Typing indicator sent to {to} for {duration}s")

        except EvolutionAPIError as e:
            # Typing indicator failures are non-critical
            logger.warning(f"Failed to send typing indicator: {e}")
        except Exception as e:
            logger.warning(f"Failed to send typing indicator: {e}")

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via Evolution API."""
        try:
            # Extract the phone number from message_id if it's in Evolution format
            # Evolution message IDs are typically in format: messageId@phonenumber
            if "@" in message_id:
                msg_id, phone = message_id.split("@", 1)
            else:
                # Fallback - use message_id as is
                msg_id = message_id
                phone = ""

            payload = {
                "readMessages": [
                    {"key": {"id": msg_id, "remoteJid": phone, "fromMe": False}}
                ]
            }

            url = self._build_url(f"markAsRead/{self.config.instance_name}")
            await self._make_request("PUT", url, payload)

            logger.debug(f"Message marked as read: {message_id}")

        except EvolutionAPIError as e:
            # Read receipt failures are non-critical
            logger.warning(f"Failed to mark message as read: {e}")
        except Exception as e:
            logger.warning(f"Failed to mark message as read: {e}")

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information via Evolution API."""
        try:
            normalized_phone = self._normalize_phone(phone)

            # Use the correct endpoint for fetching profile
            url = self._build_url(f"fetchProfile/{self.config.instance_name}")
            payload = {"number": normalized_phone}

            response_data = await self._make_request("POST", url, payload)

            if not response_data:
                return None

            contact = WhatsAppContact(
                phone=normalized_phone,
                name=response_data.get("name"),
                push_name=response_data.get("pushName"),
                profile_picture_url=response_data.get("profilePictureUrl"),
            )

            logger.debug(f"Contact info retrieved for {phone}")
            return contact

        except EvolutionAPIError as e:
            logger.warning(f"Failed to get contact info for {phone}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get contact info for {phone}: {e}")
            return None

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number."""
        try:
            normalized_phone = self._normalize_phone(phone)
            session_id = f"{self.config.instance_name}_{normalized_phone}"

            # Try to get existing session
            session = await self.session_manager.get_session(
                session_id, refresh_ttl=True
            )

            if session:
                # Update last activity
                session.last_activity = datetime.now()
                await self.session_manager.update_session(session_id, session)
                return session

            # Create new session
            contact = await self.get_contact_info(phone)
            if not contact:
                contact = WhatsAppContact(phone=normalized_phone)

            new_session = WhatsAppSession(
                session_id=session_id,
                phone_number=normalized_phone,
                contact=contact,
            )

            # Store the session
            await self.session_manager.create_session(
                session_id, new_session, ttl_seconds=self.session_ttl_seconds
            )

            logger.debug(f"Created new session for {phone}")
            return new_session

        except Exception as e:
            logger.error(f"Failed to get/create session for {phone}: {e}")
            return None

    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data."""
        try:
            session.last_activity = datetime.now()
            await self.session_manager.update_session(
                session.session_id, session, ttl_seconds=self.session_ttl_seconds
            )
            logger.debug(f"Session updated: {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to update session {session.session_id}: {e}")

    @override
    async def validate_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """Process incoming webhook data from Evolution API."""
        try:
            # Evolution API webhook structure validation
            event_type = payload.event_type
            if event_type is None:
                raise EvolutionAPIError("Event type is required in webhook payload")

            instance_name = payload.instance_id
            if instance_name is None:
                raise EvolutionAPIError("Instance name is required in webhook payload")

            if instance_name != self.config.instance_name:
                raise EvolutionAPIError(
                    f"Webhook for wrong instance: expected {self.config.instance_name}, got {instance_name}"
                )

            logger.debug(
                f"Processed webhook: {event_type} for instance {instance_name}"
            )

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to process webhook: {e}")
            raise EvolutionAPIError(f"Failed to process webhook: {e}")

    async def download_media(self, media_id: str) -> bytes:
        """Download media content by ID."""
        try:
            # Use the correct endpoint for downloading media
            url = self._build_url(
                f"getBase64FromMediaMessage/{self.config.instance_name}"
            )
            payload = {"key": {"id": media_id}}

            response_data = await self._make_request("POST", url, payload)

            if "base64" not in response_data:
                raise EvolutionAPIError("No base64 data in media response")

            import base64

            media_data = base64.b64decode(response_data["base64"])

            logger.debug(f"Media downloaded successfully: {media_id}")
            return media_data

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to download media {media_id}: {e}")
            raise EvolutionAPIError(f"Failed to download media: {e}")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        return self.config.webhook_url or ""

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        try:
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

            api_url = self._build_url(
                f"webhook/set/{self.config.instance_name}", use_message_prefix=False
            )
            await self._make_request("PUT", api_url, webhook_config)

            self.config.webhook_url = url
            logger.info(f"Webhook URL set successfully: {url}")

        except EvolutionAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to set webhook URL: {e}")
            raise EvolutionAPIError(f"Failed to set webhook URL: {e}")

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to Evolution API format.

        Evolution API expects phone numbers in the format: countrycode+number@s.whatsapp.net
        """
        # Remove non-numeric characters
        phone = "".join(c for c in phone if c.isdigit())

        # Ensure country code is present (default to Brazil +55 if not)
        # You may want to adjust this default based on your use case
        if not phone.startswith("55") and len(phone) <= 11:
            phone = "55" + phone

        # Add @s.whatsapp.net suffix if not present
        if not phone.endswith("@s.whatsapp.net"):
            phone = phone + "@s.whatsapp.net"

        return phone

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get statistics about the Evolution API provider.

        Returns:
            Dictionary with provider statistics
        """
        base_stats: Mapping[str, Any] = {
            "instance_name": self.config.instance_name,
            "base_url": self.config.base_url,
            "webhook_url": self.config.webhook_url,
            "timeout": self.config.timeout,
            "session_ttl_seconds": self.session_ttl_seconds,
        }

        # Add session manager stats
        session_stats = self.session_manager.get_stats()
        base_stats["session_stats"] = session_stats

        return base_stats
