from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import aiohttp
from rsb.functions.bytes2mime import bytes2mime

from agentle.agents.channels.models.channel_capabilities import ChannelCapabilities
from agentle.agents.channels.models.channel_media import ChannelMedia
from agentle.agents.channels.models.channel_message import ChannelMessage
from agentle.agents.channels.models.channel_send_result import ChannelSendResult
from agentle.agents.channels.models.channel_session import ChannelSession
from agentle.agents.channels.models.downloaded_channel_media import (
    DownloadedChannelMedia,
)
from agentle.agents.channels.providers.whatsapp_cloud._whatsapp_markdown import (
    to_whatsapp_markdown,
)
from agentle.agents.channels.providers.whatsapp_cloud.whatsapp_cloud_config import (
    WhatsAppCloudConfig,
)
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

logger = logging.getLogger(__name__)


class WhatsAppCloudError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class WhatsAppCloudProvider:
    """Official WhatsApp Cloud API provider for the generic channel runtime."""

    config: WhatsAppCloudConfig
    session_manager: SessionManager[ChannelSession]
    session_ttl_seconds: int
    _session: aiohttp.ClientSession | None
    _owns_session_manager: bool

    def __init__(
        self,
        config: WhatsAppCloudConfig,
        session_manager: SessionManager[ChannelSession] | None = None,
        session_ttl_seconds: int = 3600,
    ):
        self.config = config
        self.session_ttl_seconds = session_ttl_seconds
        self._session = None
        self._owns_session_manager = session_manager is None
        self.session_manager = session_manager or SessionManager(
            session_store=InMemorySessionStore[ChannelSession](),
            default_ttl_seconds=session_ttl_seconds,
        )

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_typing_indicator=False,
            supports_recording_indicator=False,
            supports_read_receipt=True,
            supports_quoting=True,
            supports_media=True,
            supports_audio=True,
            supports_markdown=False,
        )

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {
                "Authorization": f"Bearer {self.config.access_token}",
                "Content-Type": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    def get_resource_identifier(self) -> str:
        return self.config.phone_number_id

    async def initialize(self) -> None:
        if not self.config.access_token:
            raise WhatsAppCloudError("Access token is required.")
        if not self.config.phone_number_id:
            raise WhatsAppCloudError("Phone number ID is required.")
        logger.info(
            "WhatsApp Cloud provider initialized for phone_number_id=%s",
            self.config.phone_number_id,
        )

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        if self._owns_session_manager:
            await self.session_manager.close()

    def _build_url(self, endpoint: str) -> str:
        return urljoin(self.config.base_url, f"/{self.config.api_version}/{endpoint}")

    async def _handle_response(
        self, response: aiohttp.ClientResponse, expected_status: int
    ) -> Mapping[str, Any]:
        if response.status == expected_status:
            try:
                return await response.json()
            except Exception:
                return {}

        try:
            payload = await response.json()
        except Exception:
            payload = {"error": {"message": await response.text()}}

        error = payload.get("error") or {}
        message = error.get("message") or f"HTTP {response.status}"
        raise WhatsAppCloudError(message, response.status, payload)

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> Mapping[str, Any]:
        method = method.upper()
        if method == "GET":
            async with self.session.get(url) as response:
                return await self._handle_response(response, expected_status)
        if method == "POST":
            async with self.session.post(url, json=data) as response:
                return await self._handle_response(response, expected_status)
        raise ValueError(f"Unsupported HTTP method: {method}")

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        data: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._make_request(method, url, data, expected_status)
            except WhatsAppCloudError as exc:
                last_error = exc
                if exc.status_code in {400, 401, 403}:
                    raise
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        if last_error:
            raise last_error
        raise WhatsAppCloudError("Request failed.")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: str | None = None
    ) -> ChannelSendResult:
        payload: MutableMapping[str, Any] = {
            "messaging_product": "whatsapp",
            "to": self._normalize_recipient(to),
            "type": "text",
            "text": {"body": to_whatsapp_markdown(text)},
        }
        if quoted_message_id:
            payload["context"] = {"message_id": quoted_message_id}

        url = self._build_url(f"{self.config.phone_number_id}/messages")
        response = await self._make_request_with_retry("POST", url, payload)
        message_id = str((response.get("messages") or [{}])[0].get("id") or "")
        return ChannelSendResult(
            id=message_id,
            provider="whatsapp_cloud",
            resource_id=self.config.phone_number_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        filename: str | None = None,
        quoted_message_id: str | None = None,
    ) -> ChannelSendResult:
        payload: MutableMapping[str, Any] = {
            "messaging_product": "whatsapp",
            "to": self._normalize_recipient(to),
            "type": media_type,
            media_type: {"link": media_url},
        }
        if caption:
            payload[media_type]["caption"] = to_whatsapp_markdown(caption)
        if filename and media_type == "document":
            payload[media_type]["filename"] = filename
        if quoted_message_id:
            payload["context"] = {"message_id": quoted_message_id}

        url = self._build_url(f"{self.config.phone_number_id}/messages")
        response = await self._make_request_with_retry("POST", url, payload)
        message_id = str((response.get("messages") or [{}])[0].get("id") or "")
        return ChannelSendResult(
            id=message_id,
            provider="whatsapp_cloud",
            resource_id=self.config.phone_number_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_interactive_message(
        self,
        to: str,
        interactive: Mapping[str, Any],
        quoted_message_id: str | None = None,
    ) -> ChannelSendResult:
        payload: MutableMapping[str, Any] = {
            "messaging_product": "whatsapp",
            "to": self._normalize_recipient(to),
            "type": "interactive",
            "interactive": dict(interactive),
        }
        if quoted_message_id:
            payload["context"] = {"message_id": quoted_message_id}

        url = self._build_url(f"{self.config.phone_number_id}/messages")
        response = await self._make_request_with_retry("POST", url, payload)
        message_id = str((response.get("messages") or [{}])[0].get("id") or "")
        return ChannelSendResult(
            id=message_id,
            provider="whatsapp_cloud",
            resource_id=self.config.phone_number_id,
            recipient_id=to,
            raw=dict(response),
        )

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def send_recording_indicator(self, to: str, duration: int = 3) -> None:
        del to, duration

    async def mark_message_as_read(self, message_id: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        url = self._build_url(f"{self.config.phone_number_id}/messages")
        try:
            await self._make_request("POST", url, payload)
        except Exception as exc:
            logger.warning("Failed to mark message as read: %s", exc)

    async def get_session(self, contact_identifier: str) -> ChannelSession | None:
        normalized = self._normalize_contact_identifier(contact_identifier)
        session_id = f"whatsapp_cloud:{self.config.phone_number_id}:{normalized}"
        session = await self.session_manager.get_session(session_id, refresh_ttl=True)
        if session:
            session.last_activity = datetime.now()
            await self.session_manager.update_session(session_id, session)
            return session

        session = ChannelSession(
            session_id=session_id,
            contact_identifier=normalized,
        )
        created = await self.session_manager.create_session(
            session_id,
            session,
            ttl_seconds=self.session_ttl_seconds,
        )
        if not created:
            existing = await self.session_manager.get_session(
                session_id, refresh_ttl=True
            )
            if existing is not None:
                existing.last_activity = datetime.now()
                await self.session_manager.update_session(session_id, existing)
                return existing
        return session

    async def update_session(self, session: ChannelSession) -> None:
        session.last_activity = datetime.now()
        await self.session_manager.update_session(
            session.session_id,
            session,
            ttl_seconds=self.session_ttl_seconds,
        )

    async def download_media(self, media_id: str) -> DownloadedChannelMedia:
        metadata_url = self._build_url(media_id)
        metadata = await self._make_request("GET", metadata_url)
        media_url = str(metadata.get("url") or "")
        if not media_url:
            raise WhatsAppCloudError("Media URL not returned by Graph API.")

        headers = {"Authorization": f"Bearer {self.config.access_token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(media_url) as response:
                if response.status != 200:
                    raise WhatsAppCloudError(f"Failed to download media: {response.status}")
                data = await response.read()
        return DownloadedChannelMedia(data=data, mime_type=bytes2mime(data))

    @staticmethod
    def push_name_from_value(value: dict[str, Any], contact_identifier: str) -> str:
        for contact in value.get("contacts", []) or []:
            if str(contact.get("wa_id")) == contact_identifier:
                profile = contact.get("profile") or {}
                name = str(profile.get("name") or "").strip()
                if name:
                    return name
        return contact_identifier

    def parse_channel_messages(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> list[ChannelMessage]:
        del headers
        messages: list[ChannelMessage] = []
        for entry in payload.get("entry") or []:
            if not isinstance(entry, Mapping):
                continue
            for change in entry.get("changes") or []:
                if not isinstance(change, Mapping) or change.get("field") != "messages":
                    continue
                value = change.get("value") or {}
                if not isinstance(value, dict):
                    continue
                metadata = value.get("metadata") or {}
                phone_number_id = str(
                    metadata.get("phone_number_id") or self.config.phone_number_id
                )
                for message_data in value.get("messages") or []:
                    if isinstance(message_data, dict):
                        messages.append(
                            self.parse_channel_message(
                                value=value,
                                message_data=message_data,
                                phone_number_id=phone_number_id,
                            )
                        )
        return messages

    @classmethod
    def parse_channel_message(
        cls,
        *,
        value: dict[str, Any],
        message_data: dict[str, Any],
        phone_number_id: str,
    ) -> ChannelMessage:
        message_id = str(message_data.get("id") or "").strip()
        sender_id = str(message_data.get("from") or "").strip()
        if not message_id or not sender_id:
            raise ValueError("Mensagem WhatsApp Cloud sem id ou remetente.")

        timestamp = (
            datetime.fromtimestamp(int(message_data["timestamp"]))
            if message_data.get("timestamp")
            else datetime.now()
        )
        message_type = str(message_data.get("type") or "text")
        text: str | None = None
        media: ChannelMedia | None = None

        if message_type == "text":
            text = str((message_data.get("text") or {}).get("body") or "")
        elif message_type in {"image", "audio", "document", "video"}:
            media_payload = message_data.get(message_type) or {}
            media = ChannelMedia(
                media_type=message_type,
                media_id=str(media_payload.get("id") or "") or None,
                url=None,
                mime_type=str(media_payload.get("mime_type") or "") or None,
                caption=media_payload.get("caption"),
                filename=media_payload.get("filename"),
                metadata=dict(media_payload),
            )
            text = media.caption
        elif message_type == "interactive":
            interactive_payload = message_data.get("interactive") or {}
            if not isinstance(interactive_payload, dict):
                interactive_payload = {}
            interactive_type = str(interactive_payload.get("type") or "").strip()
            if interactive_type == "nfm_reply":
                reply = interactive_payload.get("nfm_reply") or {}
                if not isinstance(reply, dict):
                    reply = {}
                response_json = reply.get("response_json") or {}
                if isinstance(response_json, str):
                    try:
                        response_data = json.loads(response_json)
                    except Exception:
                        response_data = {"response_json": response_json}
                elif isinstance(response_json, dict):
                    response_data = response_json
                else:
                    response_data = {"response_json": response_json}
                text = (
                    "Resposta estruturada recebida:\n"
                    + json.dumps(response_data, ensure_ascii=False, separators=(",", ":"))
                )
            elif interactive_type == "button_reply":
                reply = interactive_payload.get("button_reply") or {}
                text = str(reply.get("title") or reply.get("id") or "").strip()
            elif interactive_type == "list_reply":
                reply = interactive_payload.get("list_reply") or {}
                text = str(reply.get("title") or reply.get("id") or "").strip()
            else:
                text = (
                    "Mensagem interativa recebida:\n"
                    + json.dumps(interactive_payload, ensure_ascii=False, separators=(",", ":"))
                )
        else:
            raise ValueError(f"Tipo de mensagem WhatsApp Cloud não suportado: {message_type}")

        return ChannelMessage(
            id=message_id,
            provider="whatsapp_cloud",
            resource_id=phone_number_id,
            conversation_id=sender_id,
            sender_id=sender_id,
            sender_display_name=cls.push_name_from_value(value, sender_id),
            message_type=message_type,
            text=text,
            media=media,
            timestamp=timestamp,
            quoted_message_id=(
                ((message_data.get("context") or {}).get("id"))
                if isinstance(message_data.get("context"), dict)
                else None
            ),
            metadata={
                "raw_message": message_data,
                "raw_value_metadata": value.get("metadata") or {},
                "interactive": message_data.get("interactive") or {},
            },
        )

    @staticmethod
    def _normalize_contact_identifier(value: str) -> str:
        return "".join(c for c in str(value) if c.isdigit()) or str(value)

    @staticmethod
    def _normalize_recipient(value: str) -> str:
        normalized = "".join(c for c in str(value) if c.isdigit())
        if normalized.startswith("55") and len(normalized) == 12:
            return normalized[:4] + "9" + normalized[4:]
        return normalized
