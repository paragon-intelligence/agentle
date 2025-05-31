from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, MutableSequence, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentle.agents.context import Context
from agentle.agents.runnable import Runnable
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.openapi.v3 import OpenAPIHandler
    from blacksheep.server.routing import MountRegistry, Router
    from rodi import ContainerProtocol


logger = logging.getLogger(__name__)


class WhatsAppBot:
    """
    WhatsApp bot that wraps an Agentle agent.

    This class handles the integration between WhatsApp messages
    and the Agentle agent, managing sessions and message conversion.
    """

    def __init__(
        self,
        agent: Runnable[Any],
        provider: WhatsAppProvider,
        config: WhatsAppBotConfig | None = None,
    ):
        """
        Initialize WhatsApp bot.

        Args:
            agent: The Agentle agent to use for processing messages
            provider: WhatsApp provider for sending/receiving messages
            config: Bot configuration
        """
        self.agent = agent
        self.provider = provider
        self.config = config or WhatsAppBotConfig()
        self._running = False
        self._webhook_handlers: MutableSequence[Callable[..., Any]] = []

    async def start(self) -> None:
        """Start the WhatsApp bot."""
        await self.provider.initialize()
        self._running = True
        logger.info("WhatsApp bot started for agent:")

    async def stop(self) -> None:
        """Stop the WhatsApp bot."""
        self._running = False
        await self.provider.shutdown()
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

    async def handle_webhook(self, payload: dict[str, Any]) -> None:
        """
        Handle incoming webhook from WhatsApp.

        Args:
            payload: Raw webhook payload
        """
        try:
            webhook_payload = await self.provider.process_webhook(payload)

            # Handle different event types
            if webhook_payload.event_type == "messages.upsert":
                await self._handle_message_upsert(webhook_payload)
            elif webhook_payload.event_type == "messages.update":
                await self._handle_message_update(webhook_payload)
            elif webhook_payload.event_type == "connection.update":
                await self._handle_connection_update(webhook_payload)

            # Call custom handlers
            for handler in self._webhook_handlers:
                await handler(webhook_payload)

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
        from blacksheep import Application, post, FromJSON, Response, json
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = Application(
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

        @post(webhook_path)
        async def _(
            webhook_payload: FromJSON[dict[str, Any]],
        ) -> Response:
            """
            Handle incoming WhatsApp webhooks.

            Args:
                webhook_payload: The webhook payload from WhatsApp

            Returns:
                Success response
            """
            try:
                # Process the webhook payload
                payload_data: dict[str, Any] = webhook_payload.value
                await self.handle_webhook(payload_data)

                # Return success response
                return json({"status": "success", "message": "Webhook processed"})

            except Exception as e:
                logger.error(f"Webhook processing error: {e}", exc_info=True)
                return json(
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
        """Convert WhatsApp message to agent input."""
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
                    FilePart(data=media_data, mime_type=message.media_mime_type)
                )

                # Add caption if present
                if message.caption:
                    parts.append(TextPart(text=message.caption))

            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                parts.append(TextPart(text="[Media file - failed to download]"))

        # Create user message
        user_message = UserMessage(parts=parts)

        # Get or create agent context
        if session.agent_context_id:
            # In a real implementation, you'd load the context from storage
            # For now, we'll create a new context
            context = Context(context_id=session.agent_context_id)
        else:
            context = Context()
            session.agent_context_id = context.context_id

        # Add message to context
        context.message_history.append(user_message)

        return context

    async def _process_with_agent(
        self, agent_input: Any, session: WhatsAppSession
    ) -> str:
        """Process input with agent and return response text."""
        try:
            # Run agent
            result = await self.agent.run_async(agent_input)

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
        data = payload.data

        # Extract message from Evolution API format
        for msg_data in data.get("messages", []):
            # Skip outgoing messages
            if msg_data.get("key", {}).get("fromMe", False):
                continue

            # Parse message
            message = self._parse_evolution_message(msg_data)
            if message:
                await self.handle_message(message)

    async def _handle_message_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle message update event (status changes)."""
        # Log status updates for debugging
        logger.debug(f"Message update: {payload.data}")

    async def _handle_connection_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle connection status update."""
        state = payload.data.get("state")
        logger.info(f"WhatsApp connection state: {state}")

    def _parse_evolution_message(
        self, msg_data: dict[str, Any]
    ) -> WhatsAppMessage | None:
        """Parse Evolution API message format."""
        try:
            key = msg_data.get("key", {})
            message_id = key.get("id")
            from_number = key.get("remoteJid")

            # Determine message type
            if "message" in msg_data:
                msg_content = msg_data["message"]

                if (
                    "conversation" in msg_content
                    or "extendedTextMessage" in msg_content
                ):
                    # Text message
                    text = msg_content.get("conversation") or msg_content.get(
                        "extendedTextMessage", {}
                    ).get("text", "")

                    return WhatsAppTextMessage(
                        id=message_id,
                        from_number=from_number,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            msg_data.get("messageTimestamp", 0)
                        ),
                        text=text,
                    )

                # Add more message type parsing as needed

        except Exception as e:
            logger.error(f"Error parsing message: {e}")

        return None
