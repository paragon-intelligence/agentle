# agentle/agents/whatsapp/providers/evolution.py
"""
Evolution API implementation for WhatsApp.
"""

import logging
import re
import time
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from typing import Any, override
from urllib.parse import urljoin

import aiohttp

from agentle.agents.whatsapp.models.downloaded_media import DownloadedMedia
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
        request_url: str | None = None,
        request_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.request_url = request_url
        self.request_data = request_data

    def __str__(self) -> str:
        """Enhanced string representation with context."""
        base_message = super().__str__()
        details: list[str] = []

        if self.status_code:
            details.append(f"status={self.status_code}")
        if self.request_url:
            details.append(f"url={self.request_url}")
        if self.response_data:
            details.append(f"response={self.response_data}")

        if details:
            return f"{base_message} ({', '.join(details)})"
        return base_message


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
        logger.info(
            "Initializing Evolution API provider with instance '%s' at %s, session_ttl=%ss",
            config.instance_name,
            config.base_url,
            session_ttl_seconds,
        )

        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._session: aiohttp.ClientSession | None = None

        # Initialize session manager
        if session_manager is None:
            logger.debug("Creating in-memory session store for Evolution API provider")
            session_store = InMemorySessionStore[WhatsAppSession]()
            self.session_manager = SessionManager(
                session_store=session_store, default_ttl_seconds=session_ttl_seconds
            )
        else:
            logger.debug("Using provided session manager for Evolution API provider")
            self.session_manager = session_manager

        logger.info(
            f"Evolution API provider initialized successfully for instance '{config.instance_name}'"
        )

    @override
    def get_instance_identifier(self) -> str:
        """Get the instance identifier for the WhatsApp provider."""
        return self.config.instance_name

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            logger.debug("Creating new aiohttp session for Evolution API")
            headers = {
                "apikey": self.config.api_key,
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
            logger.debug(f"HTTP session created with timeout={self.config.timeout}s")
        return self._session

    def _build_url(self, endpoint: str, use_message_prefix: bool = True) -> str:
        """
        Build full URL for API endpoint.

        Args:
            endpoint: The API endpoint
            use_message_prefix: Whether to prefix with /message/ (default: True)
        """
        if use_message_prefix:
            url = urljoin(self.config.base_url, f"/message/{endpoint}")
        else:
            url = urljoin(self.config.base_url, f"/{endpoint}")

        logger.debug(f"Built API URL: {url}")
        return url

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
        start_time = time.time()

        # Log request details (excluding sensitive data)
        safe_data = self._sanitize_request_data(data) if data else None
        logger.info(f"Making {method} request to {url}")
        if safe_data:
            logger.debug(f"Request payload: {safe_data}")

        try:
            match method.upper():
                case "GET":
                    async with self.session.get(url) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "POST":
                    async with self.session.post(url, json=data) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "PUT":
                    async with self.session.put(url, json=data) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case "DELETE":
                    async with self.session.delete(url) as response:
                        return await self._handle_response(
                            response, expected_status, url, data, start_time
                        )
                case _:
                    duration = time.time() - start_time
                    logger.error(
                        f"Unsupported HTTP method '{method}' for {url} (duration: {duration:.3f}s)"
                    )
                    raise ValueError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            duration = time.time() - start_time
            logger.error(
                f"HTTP client error for {method} {url} (duration: {duration:.3f}s): {type(e).__name__}: {e}",
                extra={
                    "method": method,
                    "url": url,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "request_data": self._sanitize_request_data(data) if data else None,
                },
            )
            raise EvolutionAPIError(
                f"Network error: {e}",
                request_url=url,
                request_data=self._sanitize_request_data(data) if data else None,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Unexpected error for {method} {url} (duration: {duration:.3f}s): {type(e).__name__}: {e}",
                extra={
                    "method": method,
                    "url": url,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "request_data": self._sanitize_request_data(data) if data else None,
                },
            )
            raise EvolutionAPIError(
                f"Unexpected error: {e}",
                request_url=url,
                request_data=self._sanitize_request_data(data) if data else None,
            )

    def _sanitize_request_data(
        self, data: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Remove sensitive information from request data for logging."""
        if not data:
            return data

        # Create a copy and remove sensitive fields
        sanitized = dict(data)

        # Remove API keys and tokens
        for key in list(sanitized.keys()):
            if any(
                sensitive in key.lower()
                for sensitive in ["key", "token", "secret", "password"]
            ):
                sanitized[key] = "***REDACTED***"

        return sanitized

    async def _handle_response(
        self,
        response: aiohttp.ClientResponse,
        expected_status: int,
        request_url: str,
        request_data: Mapping[str, Any] | None,
        start_time: float,
    ) -> Mapping[str, Any]:
        """
        Handle HTTP response with proper error handling.

        Args:
            response: aiohttp response object
            expected_status: Expected HTTP status code
            request_url: The URL that was requested
            request_data: The data that was sent with the request
            start_time: When the request was started

        Returns:
            Response data as dictionary

        Raises:
            EvolutionAPIError: If the response indicates an error
        """
        duration = time.time() - start_time

        logger.info(
            f"Received response {response.status} for {request_url} (duration: {duration:.3f}s)",
            extra={
                "status_code": response.status,
                "url": request_url,
                "duration_seconds": duration,
                "expected_status": expected_status,
            },
        )

        if response.status == expected_status:
            try:
                response_data = await response.json()
                logger.debug(f"Response data received: {response_data}")
                return response_data
            except Exception as e:
                logger.warning(f"Response is not valid JSON: {e}, returning empty dict")
                # If response is not JSON, return empty dict
                return {}

        # Handle error responses
        try:
            error_data = await response.json()
            logger.debug(f"Error response data: {error_data}")
        except Exception as e:
            logger.warning(f"Failed to parse error response as JSON: {e}")
            error_text = await response.text()
            error_data = {"error": error_text}
            logger.debug(f"Error response text: {error_text}")

        error_message = f"Evolution API error: {response.status}"
        if "error" in error_data:
            error_message += f" - {error_data['error']}"
        elif "message" in error_data:
            error_message += f" - {error_data['message']}"

        logger.error(
            f"API request failed: {error_message} (duration: {duration:.3f}s)",
            extra={
                "status_code": response.status,
                "url": request_url,
                "duration_seconds": duration,
                "error_data": error_data,
                "request_data": self._sanitize_request_data(request_data)
                if request_data
                else None,
            },
        )

        raise EvolutionAPIError(
            error_message,
            status_code=response.status,
            response_data=error_data,
            request_url=request_url,
            request_data=self._sanitize_request_data(request_data)
            if request_data
            else None,
        )

    async def initialize(self) -> None:
        """Initialize the Evolution API connection."""
        logger.info(
            f"Initializing Evolution API connection for instance '{self.config.instance_name}'"
        )

        try:
            # Check instance status
            url = self._build_url("instance/fetchInstances", use_message_prefix=False)
            logger.debug(f"Fetching instances from {url}")
            response_data = await self._make_request("GET", url)

            # Look for our instance in the response
            instances = (
                response_data if isinstance(response_data, list) else [response_data]
            )
            instance_found = False
            available_instances: list[str] = []

            logger.debug("Processing %d instances from API response", len(instances))

            for instance_data in instances:
                if isinstance(instance_data, dict):
                    instance_info = instance_data.get("instance", {})
                    instance_name = instance_info.get("instanceName")

                    if instance_name:
                        available_instances.append(instance_name)
                        logger.debug(f"Found instance: {instance_name}")

                        if instance_name == self.config.instance_name:
                            instance_found = True
                            logger.info(
                                f"Target instance '{self.config.instance_name}' found and accessible"
                            )

                            # Log additional instance details if available
                            if "connectionStatus" in instance_info:
                                logger.info(
                                    f"Instance connection status: {instance_info['connectionStatus']}"
                                )
                            if "profilePictureUrl" in instance_info:
                                logger.debug("Instance has profile picture configured")

            if not instance_found:
                error_msg = (
                    f"Instance '{self.config.instance_name}' not found. "
                    f"Available instances: {available_instances}"
                )
                logger.error(
                    error_msg,
                    extra={
                        "target_instance": self.config.instance_name,
                        "available_instances": available_instances,
                        "total_instances": len(available_instances),
                    },
                )
                raise EvolutionAPIError(error_msg)

            logger.info(
                f"Evolution API provider initialized successfully for instance: {self.config.instance_name}"
            )

        except EvolutionAPIError:
            logger.error("Failed to initialize Evolution API provider due to API error")
            raise
        except Exception as e:
            logger.error(
                f"Failed to initialize Evolution API provider: {type(e).__name__}: {e}",
                extra={
                    "instance_name": self.config.instance_name,
                    "base_url": self.config.base_url,
                    "error_type": type(e).__name__,
                },
            )
            raise EvolutionAPIError(f"Initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Evolution API connection."""
        logger.info("Shutting down Evolution API provider")

        try:
            if self._session:
                logger.debug("Closing aiohttp session")
                await self._session.close()
                self._session = None

            # Close session manager
            logger.debug("Closing session manager")
            await self.session_manager.close()

            logger.info("Evolution API provider shutdown complete")

        except Exception as e:
            logger.error(
                f"Error during Evolution API provider shutdown: {type(e).__name__}: {e}",
                extra={"error_type": type(e).__name__},
            )

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> WhatsAppTextMessage:
        """Send a text message via Evolution API."""
        logger.info(f"Sending text message to {to} (length: {len(text)} chars)")
        if quoted_message_id:
            logger.debug(f"Message is quoting message ID: {quoted_message_id}")

        try:
            normalized_to = self._normalize_phone(to)
            logger.debug(f"Normalized phone number: {to} -> {normalized_to}")

            payload: Mapping[str, Any] = {
                "number": normalized_to,
                "text": text,
            }

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"sendText/{self.config.instance_name}")
            response_data = await self._make_request(
                "POST", url, payload, expected_status=201
            )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            message = WhatsAppTextMessage(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                text=text,
                quoted_message_id=quoted_message_id,
            )

            logger.info(
                f"Text message sent successfully to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "text_length": len(text),
                    "has_quote": quoted_message_id is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(f"Evolution API error while sending text message to {to}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to send text message to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                    "has_quote": quoted_message_id is not None,
                },
            )
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
        logger.info(
            f"Sending {media_type} media message to {to}",
            extra={
                "to_number": to,
                "media_type": media_type,
                "media_url": media_url,
                "has_caption": caption is not None,
                "has_filename": filename is not None,
                "has_quote": quoted_message_id is not None,
            },
        )

        try:
            # Determine endpoint based on media type
            endpoint_map = {
                "image": "sendMedia",
                "document": "sendMedia",
                "audio": "sendWhatsappAudio",
                "video": "sendMedia",
            }

            endpoint = endpoint_map.get(media_type)
            if not endpoint:
                logger.error(
                    f"Unsupported media type: {media_type}. Supported types: {list(endpoint_map.keys())}"
                )
                raise EvolutionAPIError(f"Unsupported media type: {media_type}")

            normalized_to = self._normalize_phone(to)
            logger.debug(f"Normalized phone number: {to} -> {normalized_to}")
            logger.debug(f"Using endpoint: {endpoint}")

            payload: MutableMapping[str, Any] = {
                "number": normalized_to,
                "mediatype": media_type,
                "mimetype": f"{media_type}/*",
                "caption": caption or "",
                "mediaMessage": {
                    "mediaurl": media_url
                },  # Note: Evolution API uses "mediaurl" not "mediaUrl"
            }

            if caption:
                payload["mediaMessage"]["caption"] = caption
                logger.debug(f"Added caption (length: {len(caption)} chars)")

            if filename and media_type == "document":
                payload["mediaMessage"]["fileName"] = filename
                logger.debug(f"Added filename: {filename}")

            if quoted_message_id:
                payload["quoted"] = {"key": {"id": quoted_message_id}}

            url = self._build_url(f"{endpoint}/{self.config.instance_name}")
            response_data = await self._make_request(
                "POST", url, payload, expected_status=201
            )

            message_id = response_data["key"]["id"]
            from_jid = response_data["key"]["remoteJid"]

            # Create appropriate media message type
            message_class_map = {
                "image": WhatsAppImageMessage,
                "document": WhatsAppDocumentMessage,
                "audio": WhatsAppAudioMessage,
                "video": WhatsAppVideoMessage,
            }

            message_class = message_class_map[media_type]
            message = message_class(
                id=message_id,
                from_number=from_jid,
                to_number=to,
                timestamp=datetime.now(),
                status=WhatsAppMessageStatus.SENT,
                media_url=media_url,
                media_mime_type=f"{media_type}/*",
                caption=caption,
                filename=filename,
                quoted_message_id=quoted_message_id,
            )

            logger.info(
                f"{media_type.title()} media message sent successfully to {to}: {message_id}",
                extra={
                    "message_id": message_id,
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "from_jid": from_jid,
                    "media_type": media_type,
                    "media_url": media_url,
                    "has_caption": caption is not None,
                    "has_filename": filename is not None,
                },
            )
            return message

        except EvolutionAPIError:
            logger.error(
                f"Evolution API error while sending {media_type} media message to {to}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to send {media_type} media message to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "media_type": media_type,
                    "media_url": media_url,
                    "error_type": type(e).__name__,
                },
            )
            raise EvolutionAPIError(f"Failed to send media message: {e}")

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via Evolution API."""
        logger.debug(f"Sending typing indicator to {to} for {duration}s")

        try:
            normalized_to = self._normalize_phone(to)
            payload = {
                "number": normalized_to,
                "presence": "composing",
                "delay": duration * 1000,
                "options": {
                    "delay": duration * 1000,
                    "presence": "composing",
                    "number": normalized_to,
                },  # Evolution API expects milliseconds
            }

            url = self._build_url(
                f"chat/sendPresence/{self.config.instance_name}",
                use_message_prefix=False,
            )
            await self._make_request("POST", url, payload, expected_status=201)

            logger.debug(
                f"Typing indicator sent successfully to {to} for {duration}s",
                extra={
                    "to_number": to,
                    "normalized_to": normalized_to,
                    "duration_seconds": duration,
                },
            )

        except EvolutionAPIError as e:
            # Typing indicator failures are non-critical
            logger.warning(
                f"Failed to send typing indicator to {to}: {e}",
                extra={"to_number": to, "duration_seconds": duration, "error": str(e)},
            )
        except Exception as e:
            logger.warning(
                f"Failed to send typing indicator to {to}: {type(e).__name__}: {e}",
                extra={
                    "to_number": to,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                },
            )

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via Evolution API."""
        logger.debug(f"Marking message as read: {message_id}")

        try:
            # Extract the phone number from message_id if it's in Evolution format
            # Evolution message IDs are typically in format: messageId@phonenumber
            if "@" in message_id:
                msg_id, phone = message_id.split("@", 1)
                logger.debug(f"Extracted message ID: {msg_id}, phone: {phone}")
            else:
                # Fallback - use message_id as is
                msg_id = message_id
                phone = ""
                logger.debug(f"Using message ID as-is (no phone extraction): {msg_id}")

            payload = {
                "readMessages": [{"id": msg_id, "remoteJid": phone, "fromMe": False}]
            }

            url = self._build_url(
                f"chat/markMessageAsRead/{self.config.instance_name}",
                use_message_prefix=False,
            )
            await self._make_request("POST", url, payload, expected_status=201)

            logger.debug(
                f"Message marked as read successfully: {message_id}",
                extra={
                    "message_id": message_id,
                    "extracted_id": msg_id,
                    "phone": phone,
                },
            )

        except EvolutionAPIError as e:
            # Read receipt failures are non-critical
            logger.warning(
                f"Failed to mark message as read: {message_id}: {e}",
                extra={"message_id": message_id, "error": str(e)},
            )
        except Exception as e:
            logger.warning(
                f"Failed to mark message as read: {message_id}: {type(e).__name__}: {e}",
                extra={"message_id": message_id, "error_type": type(e).__name__},
            )

    async def get_contact_info(self, phone: str) -> WhatsAppContact | None:
        """Get contact information via Evolution API."""
        logger.debug(f"Fetching contact info for {phone}")

        try:
            normalized_phone = self._normalize_phone(phone)
            logger.debug(
                f"Normalized phone for contact fetch: {phone} -> {normalized_phone}"
            )

            # Use the correct endpoint for fetching profile
            url = self._build_url(
                f"chat/fetchProfile/{self.config.instance_name}",
                use_message_prefix=False,
            )

            normalized_phone = re.match(r"\d+", normalized_phone)
            normalized_phone = normalized_phone.group() if normalized_phone else ""
            payload = {"number": normalized_phone}

            response_data = await self._make_request("POST", url, payload)

            if not response_data:
                logger.debug(f"No contact data returned for {phone}")
                return None

            contact = WhatsAppContact(
                phone=normalized_phone,
                name=response_data.get("name"),
                push_name=response_data.get("pushName"),
                profile_picture_url=response_data.get("profilePictureUrl"),
            )

            logger.info(
                f"Contact info retrieved successfully for {phone}",
                extra={
                    "phone": phone,
                    "normalized_phone": normalized_phone,
                    "has_name": contact.name is not None,
                    "has_push_name": contact.push_name is not None,
                    "has_profile_picture": contact.profile_picture_url is not None,
                },
            )
            return contact

        except EvolutionAPIError as e:
            logger.warning(
                f"Evolution API error while fetching contact info for {phone}: {e}",
                extra={"phone": phone, "error": str(e)},
            )
            return None
        except Exception as e:
            logger.warning(
                f"Failed to get contact info for {phone}: {type(e).__name__}: {e}",
                extra={"phone": phone, "error_type": type(e).__name__},
            )
            return None

    async def get_session(self, phone: str) -> WhatsAppSession | None:
        """Get or create a session for a phone number."""
        logger.debug(f"Getting/creating session for {phone}")

        try:
            normalized_phone = self._normalize_phone(phone)
            session_id = f"{self.config.instance_name}_{normalized_phone}"

            logger.debug(f"Session ID: {session_id}")

            # Try to get existing session
            session = await self.session_manager.get_session(
                session_id, refresh_ttl=True
            )

            if session:
                # Update last activity
                session.last_activity = datetime.now()
                await self.session_manager.update_session(session_id, session)
                logger.debug(
                    f"Retrieved existing session for {phone}",
                    extra={
                        "phone": phone,
                        "session_id": session_id,
                        "last_activity": session.last_activity,
                    },
                )
                return session

            # Create new session
            logger.debug(f"Creating new session for {phone}")
            contact = await self.get_contact_info(phone)
            if not contact:
                logger.debug(
                    f"No contact info available, creating minimal contact for {phone}"
                )
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

            logger.info(
                f"Created new session for {phone}",
                extra={
                    "phone": phone,
                    "normalized_phone": normalized_phone,
                    "session_id": session_id,
                    "ttl_seconds": self.session_ttl_seconds,
                },
            )
            return new_session

        except Exception as e:
            logger.error(
                f"Failed to get/create session for {phone}: {type(e).__name__}: {e}",
                extra={"phone": phone, "error_type": type(e).__name__},
            )
            return None

    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data."""
        logger.debug(f"Updating session: {session.session_id}")

        try:
            session.last_activity = datetime.now()
            await self.session_manager.update_session(
                session.session_id, session, ttl_seconds=self.session_ttl_seconds
            )
            logger.debug(
                f"Session updated successfully: {session.session_id}",
                extra={
                    "session_id": session.session_id,
                    "phone_number": session.phone_number,
                    "last_activity": session.last_activity,
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to update session {session.session_id}: {type(e).__name__}: {e}",
                extra={
                    "session_id": session.session_id,
                    "phone_number": session.phone_number,
                    "error_type": type(e).__name__,
                },
            )

    @override
    async def validate_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """Process incoming webhook data from Evolution API."""
        logger.info(f"Validating webhook payload with event: {payload.event}")

        try:
            # Evolution API webhook structure validation
            event_type = payload.event
            if event_type is None:
                logger.error("Webhook validation failed: Event type is missing")
                raise EvolutionAPIError("Event type is required in webhook payload")

            instance_name = payload.instance
            if instance_name is None:
                logger.error("Webhook validation failed: Instance name is missing")
                raise EvolutionAPIError("Instance name is required in webhook payload")

            if instance_name != self.config.instance_name:
                logger.error(
                    f"Webhook validation failed: Instance mismatch - expected '{self.config.instance_name}', got '{instance_name}'"
                )
                raise EvolutionAPIError(
                    f"Webhook for wrong instance: expected {self.config.instance_name}, got {instance_name}"
                )

            logger.info(
                f"Webhook validated successfully: {event_type} for instance {instance_name}",
                extra={
                    "event_type": event_type,
                    "instance_name": instance_name,
                    "expected_instance": self.config.instance_name,
                },
            )

        except EvolutionAPIError:
            logger.error("Webhook validation failed due to Evolution API error")
            raise
        except Exception as e:
            logger.error(
                f"Failed to validate webhook: {type(e).__name__}: {e}",
                extra={"error_type": type(e).__name__},
            )
            raise EvolutionAPIError(f"Failed to process webhook: {e}")

    async def download_media(self, media_id: str) -> DownloadedMedia:
        """Download media content by ID."""
        logger.info(f"Downloading media with ID: {media_id}")

        try:
            # Use the correct endpoint for downloading media
            url = self._build_url(
                f"chat/getBase64FromMediaMessage/{self.config.instance_name}",
                use_message_prefix=False,
            )

            payload = {"message": {"key": {"id": media_id}}}

            response_data = await self._make_request(
                "POST", url, payload, expected_status=201
            )

            if "base64" not in response_data:
                logger.error(
                    f"Media download failed: No base64 data in response for media {media_id}"
                )
                raise EvolutionAPIError("No base64 data in media response")

            import base64

            media_data = base64.b64decode(response_data["base64"])
            media_size = len(media_data)

            logger.info(
                f"Media downloaded successfully: {media_id} ({media_size} bytes)",
                extra={"media_id": media_id, "size_bytes": media_size},
            )

            mimetype: str = response_data["mimetype"].split(";")[0].strip()
            return DownloadedMedia(data=media_data, mime_type=mimetype)

        except EvolutionAPIError:
            logger.error(f"Evolution API error while downloading media {media_id}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to download media {media_id}: {type(e).__name__}: {e}",
                extra={"media_id": media_id, "error_type": type(e).__name__},
            )
            raise EvolutionAPIError(f"Failed to download media: {e}")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        webhook_url = self.config.webhook_url or ""
        logger.debug(f"Retrieved webhook URL: {webhook_url}")
        return webhook_url

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        logger.info(f"Setting webhook URL: {url}")

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
            logger.info(
                f"Webhook URL set successfully: {url}",
                extra={
                    "webhook_url": url,
                    "instance_name": self.config.instance_name,
                    "events": webhook_config["webhook"]["events"],
                },
            )

        except EvolutionAPIError:
            logger.error(f"Evolution API error while setting webhook URL: {url}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to set webhook URL: {url}: {type(e).__name__}: {e}",
                extra={"webhook_url": url, "error_type": type(e).__name__},
            )
            raise EvolutionAPIError(f"Failed to set webhook URL: {e}")

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to Evolution API format.

        Evolution API expects phone numbers in the format: countrycode+number@s.whatsapp.net
        """
        original_phone = phone

        # Remove non-numeric characters
        phone = "".join(c for c in phone if c.isdigit())

        # Ensure country code is present (default to Brazil +55 if not)
        # You may want to adjust this default based on your use case
        if not phone.startswith("55") and len(phone) <= 11:
            phone = "55" + phone

        # Add @s.whatsapp.net suffix if not present
        if not phone.endswith("@s.whatsapp.net"):
            phone = phone + "@s.whatsapp.net"

        if original_phone != phone:
            logger.debug(f"Phone number normalized: {original_phone} -> {phone}")

        return phone

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get statistics about the Evolution API provider.

        Returns:
            Dictionary with provider statistics
        """
        logger.debug("Retrieving provider statistics")

        base_stats: Mapping[str, Any] = {
            "instance_name": self.config.instance_name,
            "base_url": self.config.base_url,
            "webhook_url": self.config.webhook_url,
            "timeout": self.config.timeout,
            "session_ttl_seconds": self.session_ttl_seconds,
            "has_active_session": self._session is not None,
        }

        # Add session manager stats
        session_stats = self.session_manager.get_stats()
        base_stats["session_stats"] = session_stats

        logger.debug(f"Provider statistics: {base_stats}")
        return base_stats
