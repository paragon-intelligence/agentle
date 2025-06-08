from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, MutableSequence, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from rsb.coroutines.run_sync import run_sync

from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_protocol import AgentProtocol
from agentle.agents.context import Context
from agentle.agents.whatsapp.models.data import Data
from agentle.agents.whatsapp.models.message import Message
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.openapi.v3 import OpenAPIHandler
    from blacksheep.server.routing import MountRegistry, Router
    from rodi import ContainerProtocol

try:
    import blacksheep
except ImportError:
    pass

logger = logging.getLogger(__name__)


class WhatsAppBot:
    """
    WhatsApp bot that wraps an Agentle agent.

    This class handles the integration between WhatsApp messages
    and the Agentle agent, managing sessions and message conversion.
    """

    agent: AgentProtocol[Any]
    provider: WhatsAppProvider
    config: WhatsAppBotConfig
    context_manager: SessionManager[Context]
    _running: bool
    _webhook_handlers: MutableSequence[Callable[..., Any]]

    def __init__(
        self,
        agent: AgentProtocol[Any],
        provider: WhatsAppProvider,
        config: WhatsAppBotConfig | None = None,
        context_manager: SessionManager[Context] | None = None,
    ):
        """
        Initialize WhatsApp bot.

        Args:
            agent: The Agentle agent to use for processing messages
            provider: WhatsApp provider for sending/receiving messages
            config: Bot configuration
            context_manager: Session manager for conversation context persistence (creates in-memory if not provided)
        """
        self.agent = agent
        self.provider = provider
        self.config = config or WhatsAppBotConfig()
        self._running = False
        self._webhook_handlers: MutableSequence[Callable[..., Any]] = []

        # Initialize context manager using existing session system
        if context_manager is None:
            context_store = InMemorySessionStore[Context]()
            self.context_manager = SessionManager(
                session_store=context_store,
                default_ttl_seconds=1800,  # 30 minutes default for conversations
            )
        else:
            self.context_manager = context_manager

    def startt(self) -> None:
        """Start the WhatsApp bot."""
        run_sync(self.start_async)

    def stopp(self) -> None:
        """Stop the WhatsApp bot."""
        run_sync(self.stop_async)

    async def start_async(self) -> None:
        """Start the WhatsApp bot."""
        await self.provider.initialize()
        self._running = True
        logger.info("WhatsApp bot started for agent:")

    async def stop_async(self) -> None:
        """Stop the WhatsApp bot."""
        self._running = False
        await self.provider.shutdown()
        await self.context_manager.close()
        logger.info("WhatsApp bot stopped for agent:")

    async def handle_message(self, message: WhatsAppMessage) -> None:
        """
        Handle incoming WhatsApp message.

        Args:
            message: The incoming WhatsApp message
        """
        try:
            # Mark as read if configured
            if self.config.auto_read_messages:
                await self.provider.mark_message_as_read(message.id)

            # Get or create session
            session = await self.provider.get_session(message.from_number)
            if not session:
                logger.error(f"Failed to get session for {message.from_number}")
                return

            # Check if this is first interaction
            if session.message_count == 0 and self.config.welcome_message:
                await self.provider.send_text_message(
                    message.from_number, self.config.welcome_message
                )

            # Show typing indicator
            if self.config.typing_indicator:
                await self.provider.send_typing_indicator(
                    message.from_number, self.config.typing_duration
                )

            # Convert WhatsApp message to agent input
            agent_input = await self._convert_message_to_input(message, session)

            # Process with agent
            response = await self._process_with_agent(agent_input, session)

            # Send response
            await self._send_response(message.from_number, response, message.id)

            # Update session
            session.message_count += 1
            session.last_activity = datetime.now()
            await self.provider.update_session(session)

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self._send_error_message(message.from_number, message.id)

    async def handle_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """
        Handle incoming webhook from WhatsApp.

        Args:
            payload: Raw webhook payload
        """
        try:
            await self.provider.validate_webhook(payload)

            # Handle Evolution API events
            if payload.event == "messages.upsert":
                await self._handle_message_upsert(payload)
            elif payload.event == "messages.update":
                await self._handle_message_update(payload)
            elif payload.event == "connection.update":
                await self._handle_connection_update(payload)
            # Handle Meta API events
            elif payload.entry:
                await self._handle_meta_webhook(payload)

            # Call custom handlers
            for handler in self._webhook_handlers:
                await handler(payload)

        except Exception as e:
            logger.error(f"Error handling webhook: {e}", exc_info=True)

    def to_blacksheep_app(
        self,
        *,
        router: Router | None = None,
        services: ContainerProtocol | None = None,
        show_error_details: bool = False,
        mount: MountRegistry | None = None,
        docs: OpenAPIHandler | None = None,
        webhook_path: str = "/webhook/whatsapp",
    ) -> Application:
        """
        Convert the WhatsApp bot to a BlackSheep ASGI application.

        Args:
            router: Optional router to use
            services: Optional services container
            show_error_details: Whether to show error details in responses
            mount: Optional mount registry
            docs: Optional OpenAPI handler
            webhook_path: Path for the webhook endpoint

        Returns:
            BlackSheep application with webhook endpoint
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application(
            router=router,
            services=services,
            show_error_details=show_error_details,
            mount=mount,
        )

        if docs is None:
            docs = OpenAPIHandler(
                ui_path="/openapi",
                info=Info(title="Agentle WhatsApp Bot API", version="1.0.0"),
            )
            docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))

        docs.bind_app(app)

        @blacksheep.post(webhook_path)
        async def _(
            webhook_payload: blacksheep.FromJSON[WhatsAppWebhookPayload],
        ) -> blacksheep.Response:
            """
            Handle incoming WhatsApp webhooks.

            Args:
                webhook_payload: The webhook payload from WhatsApp

            Returns:
                Success response
            """
            try:
                # Process the webhook payload
                payload_data: WhatsAppWebhookPayload = webhook_payload.value
                await self.handle_webhook(payload_data)

                # Return success response
                return blacksheep.json(
                    {"status": "success", "message": "Webhook processed"}
                )

            except Exception as e:
                logger.error(f"Webhook processing error: {e}", exc_info=True)
                return blacksheep.json(
                    {"status": "error", "message": "Failed to process webhook"},
                    status=500,
                )

        return app

    def add_webhook_handler(self, handler: Callable[..., Any]) -> None:
        """Add custom webhook handler."""
        self._webhook_handlers.append(handler)

    async def _convert_message_to_input(
        self, message: WhatsAppMessage, session: WhatsAppSession
    ) -> Any:
        """Convert WhatsApp message to agent input with proper context loading."""
        parts: MutableSequence[TextPart | FilePart] = []

        # Handle text messages
        if isinstance(message, WhatsAppTextMessage):
            parts.append(TextPart(text=message.text))

        # Handle media messages
        elif isinstance(message, WhatsAppMediaMessage):
            # Download media
            try:
                media_data = await self.provider.download_media(message.id)
                parts.append(
                    FilePart(data=media_data.data, mime_type=media_data.mime_type)
                )

                # Add caption if present
                if message.caption:
                    parts.append(TextPart(text=f"Caption: {message.caption}"))

            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                parts.append(TextPart(text="[Media file - failed to download]"))

        # Create user message
        user_message = UserMessage.create_named(parts=parts, name=message.push_name)

        # Get or create agent context with proper persistence
        context: Context
        if session.agent_context_id:
            # Load existing context from storage
            existing_context = await self.context_manager.get_session(
                session.agent_context_id, refresh_ttl=True
            )
            if existing_context:
                context = existing_context
                logger.debug(f"Loaded existing context: {session.agent_context_id}")
            else:
                # Context expired or not found, create new one
                context = Context(context_id=session.agent_context_id)
                logger.debug(
                    f"Context not found, created new: {session.agent_context_id}"
                )
        else:
            # Create new context
            context = Context()
            session.agent_context_id = context.context_id
            logger.debug(f"Created new context: {context.context_id}")

        # Add message to context
        context.message_history.append(user_message)

        # Save context to storage
        await self.context_manager.update_session(
            context.context_id, context, create_if_missing=True
        )

        return context

    async def _process_with_agent(
        self, agent_input: AgentInput, session: WhatsAppSession
    ) -> str:
        """Process input with agent and return response text."""
        try:
            async with self.agent.start_mcp_servers_async():
                # Run agent with the full context
                result = await self.agent.run_async(agent_input)

            # Save the updated context after agent processing
            if result.context and hasattr(agent_input, "context_id"):
                await self.context_manager.update_session(
                    cast(Context, agent_input).context_id,
                    result.context,  # The updated context from agent execution
                    create_if_missing=True,
                )
                logger.debug(
                    f"Saved updated context: {cast(Context, agent_input).context_id}"
                )

            if result.generation:
                return result.text

            return "I processed your message but have no response."

        except Exception as e:
            logger.error(f"Agent processing error: {e}", exc_info=True)
            raise

    async def _send_response(
        self, to: str, response: str, reply_to: str | None = None
    ) -> None:
        """Send response message(s) to user."""
        # Split long messages
        messages = self._split_message(response)

        for i, msg in enumerate(messages):
            # Only quote the first message
            quoted_id = reply_to if i == 0 else None

            await self.provider.send_text_message(
                to=to, text=msg, quoted_message_id=quoted_id
            )

            # Small delay between messages
            if i < len(messages) - 1:
                await asyncio.sleep(0.5)

    async def _send_error_message(self, to: str, reply_to: str | None = None) -> None:
        """Send error message to user."""
        await self.provider.send_text_message(
            to=to, text=self.config.error_message, quoted_message_id=reply_to
        )

    def _split_message(self, text: str) -> Sequence[str]:
        """Split long message into chunks."""
        if len(text) <= self.config.max_message_length:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        messages: MutableSequence[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.config.max_message_length:
                if current:
                    current += "\n\n"
                current += para
            else:
                if current:
                    messages.append(current)
                current = para

        if current:
            messages.append(current)

        # Further split if any message is still too long
        final_messages = []
        for msg in messages:
            if len(msg) <= self.config.max_message_length:
                final_messages.append(msg)
            else:
                # Hard split
                for i in range(0, len(msg), self.config.max_message_length):
                    final_messages.append(msg[i : i + self.config.max_message_length])

        return final_messages

    async def _handle_message_upsert(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle new message event."""
        # Check if this is Evolution API format
        if payload.event == "messages.upsert" and payload.data:
            # Evolution API format - single message in data field
            data = payload.data

            # Skip outgoing messages
            if data["key"].get("fromMe", False):
                return

            # Parse message directly from data (which contains the message info)
            message = self._parse_evolution_message_from_data(data)
            if message:
                await self.handle_message(message)

        # Check if this is Meta API format
        elif payload.entry:
            # Meta API format - handle through provider
            await self.provider.validate_webhook(payload)
            # Meta API provider should handle message parsing differently
            # For now, we'll delegate this to the provider
            pass
        else:
            logger.warning("Unknown webhook format in message upsert")

    async def _handle_message_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle message update event (status changes)."""
        if payload.event == "messages.update" and payload.data:
            # Evolution API format
            logger.debug(f"Message update: {payload.data}")
        elif payload.entry:
            # Meta API format
            logger.debug(f"Message update: {payload.entry}")
        else:
            logger.debug(f"Message update: {payload}")

    async def _handle_connection_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle connection status update."""
        if payload.event == "connection.update" and payload.data:
            # Evolution API format
            # Note: connection updates might have different data structure
            logger.info(f"WhatsApp connection update: {payload.data}")
        elif payload.entry:
            # Meta API format
            logger.info(f"WhatsApp connection update: {payload.entry}")
        else:
            logger.info(f"WhatsApp connection update: {payload}")

    def _parse_evolution_message_from_data(self, data: Data) -> WhatsAppMessage | None:
        """Parse Evolution API message from webhook data field."""
        try:
            # Extract key information
            key = data["key"]
            message_id = key.get("id")
            from_number = key.get("remoteJid")

            # Check if there's a message field
            if data.get("message"):
                msg_content = cast(Message, data.get("message"))

                # Handle text messages
                if msg_content.get("conversation"):
                    text = msg_content.get("conversation")
                    return WhatsAppTextMessage(
                        id=message_id,
                        push_name=data["pushName"],
                        from_number=from_number,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0)
                            / 1000  # Convert from milliseconds
                        ),
                        text=text or ".",
                    )

                # Handle extended text messages
                elif msg_content.get("extendedTextMessage"):
                    extended_text_message = msg_content.get("extendedTextMessage")
                    text = (
                        extended_text_message.get("text", "")
                        if extended_text_message
                        else ""
                    )
                    return WhatsAppTextMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data["pushName"],
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0) / 1000
                        ),
                        text=text,
                    )

                # Handle image messages
                elif msg_content.get("imageMessage"):
                    image_msg = msg_content.get("imageMessage")
                    return WhatsAppImageMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data["pushName"],
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0) / 1000
                        ),
                        media_url=image_msg.get("url", "") if image_msg else "",
                        media_mime_type=image_msg.get("mimetype", "image/jpeg")
                        if image_msg
                        else "image/jpeg",
                        caption=image_msg.get("caption") if image_msg else "",
                    )

                # Handle document messages
                elif msg_content.get("documentMessage"):
                    doc_msg = msg_content.get("documentMessage")
                    return WhatsAppDocumentMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data["pushName"],
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0) / 1000
                        ),
                        media_url=doc_msg.get("url", "") if doc_msg else "",
                        media_mime_type=doc_msg.get(
                            "mimetype", "application/octet-stream"
                        )
                        if doc_msg
                        else "application/octet-stream",
                        filename=doc_msg.get("fileName") if doc_msg else "",
                        caption=doc_msg.get("caption") if doc_msg else "",
                    )

                # Handle audio messages
                elif msg_content.get("audioMessage"):
                    audio_msg = msg_content.get("audioMessage")
                    return WhatsAppAudioMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data["pushName"],
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0) / 1000
                        ),
                        media_url=audio_msg.get("url", "") if audio_msg else "",
                        media_mime_type=audio_msg.get("mimetype", "audio/ogg")
                        if audio_msg
                        else "audio/ogg",
                    )
                elif msg_content.get("videoMessage"):
                    video_msg = msg_content.get("videoMessage")
                    return WhatsAppVideoMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data["pushName"],
                        caption=video_msg.get("caption") if video_msg else None,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            getattr(data, "messageTimestamp", 0) / 1000
                        ),
                        media_url=video_msg.get("url", "") if video_msg else "",
                        media_mime_type=video_msg.get("mimetype", "")
                        if video_msg
                        else "",
                    )

        except Exception as e:
            logger.error(f"Error parsing Evolution message from data: {e}")

        return None

    async def _handle_meta_webhook(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle Meta WhatsApp Business API webhooks."""
        try:
            if not payload.entry:
                return

            for entry_item in payload.entry:
                changes = entry_item.get("changes", [])
                for change in changes:
                    field = change.get("field")
                    value = change.get("value", {})

                    if field == "messages":
                        # Process incoming messages
                        messages = value.get("messages", [])
                        for msg_data in messages:
                            # Skip outgoing messages
                            if (
                                msg_data.get("from")
                                == self.provider.get_instance_identifier()
                            ):
                                continue

                            message = await self._parse_meta_message(msg_data)
                            if message:
                                await self.handle_message(message)

        except Exception as e:
            logger.error(f"Error handling Meta webhook: {e}")

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
                    push_name=msg_data.get(
                        "pushName",  # TODO(arthur): check Meta's official API
                        "user",
                    ),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    text=text,
                )

            elif msg_type == "image":
                image_data = msg_data.get("image", {})

                return WhatsAppImageMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get(
                        "pushName",  # TODO(arthur): check Meta's official API
                        "user",
                    ),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=image_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=image_data.get("mime_type", "image/jpeg"),
                    caption=image_data.get("caption"),
                )

            elif msg_type == "document":
                doc_data = msg_data.get("document", {})

                return WhatsAppDocumentMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get(
                        "pushName",  # TODO(arthur): check Meta's official API
                        "user",
                    ),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=doc_data.get("id", ""),  # Meta uses ID for media
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
                    push_name=msg_data.get(
                        "pushName",  # TODO(arthur): check Meta's official API
                        "user",
                    ),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=audio_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=audio_data.get("mime_type", "audio/ogg"),
                )

        except Exception as e:
            logger.error(f"Error parsing Meta message: {e}")

        return None
