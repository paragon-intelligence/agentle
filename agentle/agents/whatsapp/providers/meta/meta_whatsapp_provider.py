"""
Meta WhatsApp Business API implementation.
"""

import asyncio
import hashlib
import hmac
import logging
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from typing import Any, override
from urllib.parse import urljoin

import aiohttp
from rsb.functions.bytes2mime import bytes2mime

from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_message_status import WhatsAppMessageStatus
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.meta.meta_whatsapp_config import (
    MetaWhatsAppConfig,
)
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MetaWhatsAppError(Exception):
    """Exception raised for Meta WhatsApp Business API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: int | None = None,
        error_subcode: int | None = None,
        response_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_subcode = error_subcode
        self.response_data = response_data


class MetaWhatsAppProvider(WhatsAppProvider):
    """
    Meta WhatsApp Business API implementation.

    This provider implements the WhatsApp interface using Meta's official
    WhatsApp Business API, providing enterprise-grade features and reliability.

    Features:
    - Official Meta WhatsApp Business API integration
    - Automatic session management with configurable storage
    - Webhook signature verification
    - Comprehensive error handling and retry logic
    - Media handling with automatic uploads
    - Message templates support
    - Status tracking and delivery receipts
    """

    config: MetaWhatsAppConfig
    session_manager: SessionManager[WhatsAppSession]
    session_ttl_seconds: int
    _session: aiohttp.ClientSession | None

    def __init__(
        self,
        config: MetaWhatsAppConfig,
        session_manager: SessionManager[WhatsAppSession] | None = None,
        session_ttl_seconds: int = 3600,
    ):
        """
        Initialize Meta WhatsApp Business provider.

        Args:
            config: Meta WhatsApp Business API configuration
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
        return self.config.phone_number_id

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            headers = {
                "Authorization": f"Bearer {self.config.access_token}",
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL for API endpoint.

        Args:
            endpoint: The API endpoint
        """
        return urljoin(self.config.base_url, f"/{self.config.api_version}/{endpoint}")

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> Mapping[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Full URL for the request
            data: Optional JSON data to send
            expected_status: Expected HTTP status code

        Returns:
            Response data as dictionary

        Raises:
            MetaWhatsAppError: If the request fails after all retries
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._make_request(method, url, data, expected_status)
            except MetaWhatsAppError as e:
                last_exception = e

                # Don't retry for authentication errors or invalid requests
                if e.status_code in [401, 403, 400]:
                    raise

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    logger.warning(
                        f"Request failed, retrying (attempt {attempt + 1}): {e}"
                    )

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise MetaWhatsAppError("Request failed after all retries")

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
            MetaWhatsAppError: If the request fails
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
            raise MetaWhatsAppError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise MetaWhatsAppError(f"Unexpected error: {e}")

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
            MetaWhatsAppError: If the response indicates an error
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
            error_data = {"error": {"message": await response.text()}}

        error_info = error_data.get("error", {})
        error_message = error_info.get("message", f"HTTP {response.status}")

        # Safely convert error codes to int
        error_code = None
        error_subcode = None

        if "code" in error_info:
            try:
                error_code = int(error_info["code"])
            except (ValueError, TypeError):
                pass

        if "error_subcode" in error_info:
            try:
                error_subcode = int(error_info["error_subcode"])
            except (ValueError, TypeError):
                pass

        logger.error(f"Meta WhatsApp API error: {error_message}")
        raise MetaWhatsAppError(
            error_message,
            status_code=response.status,
            error_code=error_code,
            error_subcode=error_subcode,
            response_data=error_data,
        )

    async def initialize(self) -> None:
        """Initialize the Meta WhatsApp Business API connection."""
        try:
            # Verify access token by getting business account info
            url = self._build_url(f"{self.config.business_account_id}")
            response_data = await self._make_request("GET", url)

            account_name = response_data.get("name", "Unknown")
            logger.info(
                f"Meta WhatsApp provider initialized for account: {account_name} "
                + f"(Phone ID: {self.config.phone_number_id})"
            )

        except MetaWhatsAppError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Meta WhatsApp provider: {e}")
            raise MetaWhatsAppError(f"Initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Meta WhatsApp Business API connection."""
        try:
            if self._session:
                await self._session.close()
                self._session = None

            # Close session manager
            await self.session_manager.close()

            logger.info("Meta WhatsApp provider shutdown complete")

        except Exception as e:
            logger.error(f"Error during Meta WhatsApp provider shutdown: {e}")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """Send a text message via Meta WhatsApp Business API."""
        try:
            payload: MutableMapping[str, Any] = {
                "messaging_product": "whatsapp",
                "to": self._normalize_phone(to),
                "type": "text",
                "text": {"body": text},
            }

            if quoted_message_id:
                payload["context"] = {"message_id": quoted_message_id}

            url = self._build_url(f"{self.config.phone_number_id}/messages")
            response_data = await self._make_request_with_retry("POST", url, payload)

            message = WhatsAppTextMessage(
                id=response_data["messages"][0]["id"],
                from_number=self.config.phone_number_id,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                text=text,
                quoted_message_id=quoted_message_id,
            )

            logger.debug(f"Text message sent successfully: {message.id}")
            return message

        except MetaWhatsAppError:
            raise
        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
            raise MetaWhatsAppError(f"Failed to send text message: {e}")

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> WhatsAppMediaMessage:
        """Send a media message via Meta WhatsApp Business API."""
        try:
            # First upload media to get media ID
            media_id = await self._upload_media(media_url, media_type)

            payload: MutableMapping[str, Any] = {
                "messaging_product": "whatsapp",
                "to": self._normalize_phone(to),
                "type": media_type,
                media_type: {"id": media_id},
            }

            if caption:
                payload[media_type]["caption"] = caption

            if filename and media_type == "document":
                payload[media_type]["filename"] = filename

            if quoted_message_id:
                payload["context"] = {"message_id": quoted_message_id}

            url = self._build_url(f"{self.config.phone_number_id}/messages")
            response_data = await self._make_request_with_retry("POST", url, payload)

            # Create appropriate media message type
            message_class_map = {
                "image": WhatsAppImageMessage,
                "document": WhatsAppDocumentMessage,
                "audio": WhatsAppAudioMessage,
                "video": WhatsAppVideoMessage,
            }

            message_class = message_class_map[media_type]
            message = message_class(
                id=response_data["messages"][0]["id"],
                from_number=self.config.phone_number_id,
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

        except MetaWhatsAppError:
            raise
        except Exception as e:
            logger.error(f"Failed to send media message: {e}")
            raise MetaWhatsAppError(f"Failed to send media message: {e}")

    async def _upload_media(self, media_url: str, media_type: str) -> str:
        """Upload media to Meta and return media ID."""
        try:
            # Download the media first
            async with aiohttp.ClientSession() as session:
                async with session.get(media_url) as response:
                    if response.status != 200:
                        raise MetaWhatsAppError(
                            f"Failed to download media: {response.status}"
                        )
                    media_data = await response.read()

            # Upload to Meta
            upload_url = self._build_url(f"{self.config.phone_number_id}/media")

            form_data = aiohttp.FormData()
            form_data.add_field("messaging_product", "whatsapp")
            form_data.add_field("type", media_type)
            form_data.add_field(
                "file",
                media_data,
                filename=f"media.{media_type}",
                content_type=f"{media_type}/*",
            )

            # Create a separate session for file upload (different headers)
            headers = {"Authorization": f"Bearer {self.config.access_token}"}
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            async with aiohttp.ClientSession(
                headers=headers, timeout=timeout
            ) as upload_session:
                async with upload_session.post(upload_url, data=form_data) as response:
                    response_data = await self._handle_response(response, 200)
                    return response_data["id"]

        except Exception as e:
            logger.error(f"Failed to upload media: {e}")
            raise MetaWhatsAppError(f"Failed to upload media: {e}")

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via Meta WhatsApp Business API."""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "to": self._normalize_phone(to),
                "type": "typing",
            }

            url = self._build_url(f"{self.config.phone_number_id}/messages")
            await self._make_request("POST", url, payload)

            logger.debug(f"Typing indicator sent to {to}")

        except MetaWhatsAppError as e:
            # Typing indicator failures are non-critical
            logger.warning(f"Failed to send typing indicator: {e}")
        except Exception as e:
            logger.warning(f"Failed to send typing indicator: {e}")

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via Meta WhatsApp Business API."""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            }

            url = self._build_url(f"{self.config.phone_number_id}/messages")
            await self._make_request("POST", url, payload)

            logger.debug(f"Message marked as read: {message_id}")

        except MetaWhatsAppError as e:
            # Read receipt failures are non-critical
            logger.warning(f"Failed to mark message as read: {e}")
        except Exception as e:
            logger.warning(f"Failed to mark message as read: {e}")

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information via Meta WhatsApp Business API."""
        try:
            normalized_phone = self._normalize_phone(phone)

            # Meta doesn't provide a direct contact info endpoint
            # We'll create a basic contact object
            contact = WhatsAppContact(
                phone=normalized_phone,
                name=None,
                push_name=None,
                profile_picture_url=None,
            )

            logger.debug(f"Contact info created for {phone}")
            return contact

        except Exception as e:
            logger.warning(f"Failed to get contact info for {phone}: {e}")
            return None

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number."""
        try:
            normalized_phone = self._normalize_phone(phone)
            session_id = f"{self.config.phone_number_id}_{normalized_phone}"

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
        """Process incoming webhook data from Meta WhatsApp Business API."""
        try:
            # Verify webhook signature if needed
            # TODO(arthur): verify the webhook signature

            # Meta webhook structure validation
            entry = payload.entry
            if not entry:
                raise MetaWhatsAppError("No entry data in webhook payload")

            # Process each entry
            for entry_item in entry:
                changes = entry_item.get("changes", [])
                if not changes:
                    continue

                for change in changes:
                    field = change.get("field")
                    value = change.get("value", {})

                    if field == "messages":
                        # Process incoming messages
                        await self._process_messages_webhook(value)
                    elif field == "message_status":
                        # Process message status updates
                        await self._process_status_webhook(value)

            logger.debug(f"Processed webhook for phone {self.config.phone_number_id}")

        except MetaWhatsAppError:
            raise
        except Exception as e:
            logger.error(f"Failed to process webhook: {e}")
            raise MetaWhatsAppError(f"Failed to process webhook: {e}")

    async def _process_messages_webhook(self, value: dict[str, Any]) -> None:
        """Process incoming messages from Meta webhook."""
        try:
            messages = value.get("messages", [])
            for msg_data in messages:
                # Skip if not a user message
                if msg_data.get("from") == self.config.phone_number_id:
                    continue

                message = await self._parse_meta_message(msg_data)
                if message:
                    # Create a bot instance or get existing one to handle message
                    # This would need to be coordinated with WhatsAppBot
                    logger.info(f"Received message from Meta API: {message.id}")

        except Exception as e:
            logger.error(f"Error processing messages webhook: {e}")

    async def _process_status_webhook(self, value: dict[str, Any]) -> None:
        """Process message status updates from Meta webhook."""
        try:
            statuses = value.get("statuses", [])
            for status_data in statuses:
                message_id = status_data.get("id")
                status = status_data.get("status")
                logger.debug(f"Message {message_id} status: {status}")

        except Exception as e:
            logger.error(f"Error processing status webhook: {e}")

    async def _parse_meta_message(
        self, msg_data: dict[str, Any]
    ) -> WhatsAppMessage | None:
        """Parse Meta API message format."""
        try:
            message_id = msg_data.get("id")
            from_number = msg_data.get("from")
            timestamp_str = msg_data.get("timestamp")

            if not message_id or not from_number:
                return None

            # Convert timestamp
            timestamp = (
                datetime.fromtimestamp(int(timestamp_str))
                if timestamp_str
                else datetime.now()
            )

            # Handle different message types
            msg_type = msg_data.get("type")

            if msg_type == "text":
                text_data = msg_data.get("text", {})
                text = text_data.get("body", "")

                return WhatsAppTextMessage(
                    id=message_id,
                    from_number=from_number,
                    to_number=self.config.phone_number_id,
                    timestamp=timestamp,
                    text=text,
                )

            elif msg_type == "image":
                image_data = msg_data.get("image", {})

                return WhatsAppImageMessage(
                    id=message_id,
                    from_number=from_number,
                    to_number=self.config.phone_number_id,
                    timestamp=timestamp,
                    media_url="",  # Will be downloaded later
                    media_mime_type=image_data.get("mime_type", "image/jpeg"),
                    caption=image_data.get("caption"),
                )

            elif msg_type == "document":
                doc_data = msg_data.get("document", {})

                return WhatsAppDocumentMessage(
                    id=message_id,
                    from_number=from_number,
                    to_number=self.config.phone_number_id,
                    timestamp=timestamp,
                    media_url="",  # Will be downloaded later
                    media_mime_type=doc_data.get(
                        "mime_type", "application/octet-stream"
                    ),
                    filename=doc_data.get("filename"),
                    caption=doc_data.get("caption"),
                )

            elif msg_type == "audio":
                audio_data = msg_data.get("audio", {})

                return WhatsAppAudioMessage(
                    id=message_id,
                    from_number=from_number,
                    to_number=self.config.phone_number_id,
                    timestamp=timestamp,
                    media_url="",  # Will be downloaded later
                    media_mime_type=audio_data.get("mime_type", "audio/ogg"),
                )

        except Exception as e:
            logger.error(f"Error parsing Meta message: {e}")

        return None

    def verify_webhook_signature(self, payload_body: str, signature: str) -> bool:
        """
        Verify webhook signature for security.

        Args:
            payload_body: Raw webhook payload body
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Remove 'sha256=' prefix if present
            if signature.startswith("sha256="):
                signature = signature[7:]

            # Calculate expected signature
            expected_signature = hmac.new(
                self.config.app_secret.encode(),
                payload_body.encode(),
                hashlib.sha256,
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.error(f"Failed to verify webhook signature: {e}")
            return False

    async def download_media(self, media_id: str) -> DownloadedMedia:
        """Download media content by ID."""
        try:
            # First get media URL
            url = self._build_url(media_id)
            response_data = await self._make_request("GET", url)

            media_url = response_data.get("url")
            if not media_url:
                raise MetaWhatsAppError("No media URL in response")

            # Download media content
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.config.access_token}"}
                async with session.get(media_url, headers=headers) as response:
                    if response.status != 200:
                        raise MetaWhatsAppError(
                            f"Failed to download media: {response.status}"
                        )
                    media_data = await response.read()

            logger.debug(f"Media downloaded successfully: {media_id}")
            return DownloadedMedia(data=media_data, mime_type=bytes2mime(media_data))

        except MetaWhatsAppError:
            raise
        except Exception as e:
            logger.error(f"Failed to download media {media_id}: {e}")
            raise MetaWhatsAppError(f"Failed to download media: {e}")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        return self.config.webhook_url or ""

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        try:
            # Note: In Meta WhatsApp Business API, webhook URLs are typically set
            # through the Facebook Developer Console, not via API
            # This method updates the local config
            self.config.webhook_url = url
            logger.info(f"Webhook URL updated in config: {url}")

        except Exception as e:
            logger.error(f"Failed to set webhook URL: {e}")
            raise MetaWhatsAppError(f"Failed to set webhook URL: {e}")

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to Meta WhatsApp Business API format.

        Meta expects phone numbers in international format without + prefix.
        """
        # Remove non-numeric characters
        phone = "".join(c for c in phone if c.isdigit())

        # Remove leading + if present
        if phone.startswith("+"):
            phone = phone[1:]

        # Ensure country code is present # TODO(arthur): change this logic
        if len(phone) <= 10:  # Assuming it's missing country code
            phone = (
                "1" + phone
            )  # Default to US country code # TODO(arthur): change this logic

        return phone

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get statistics about the Meta WhatsApp provider.

        Returns:
            Dictionary with provider statistics
        """
        base_stats: Mapping[str, Any] = {
            "phone_number_id": self.config.phone_number_id,
            "business_account_id": self.config.business_account_id,
            "api_version": self.config.api_version,
            "base_url": self.config.base_url,
            "webhook_url": self.config.webhook_url,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "session_ttl_seconds": self.session_ttl_seconds,
        }

        # Add session manager stats
        session_stats = self.session_manager.get_stats()
        base_stats["session_stats"] = session_stats

        return base_stats
