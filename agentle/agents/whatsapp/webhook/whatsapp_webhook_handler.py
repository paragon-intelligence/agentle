from typing import TYPE_CHECKING, Any
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
import logging

if TYPE_CHECKING:
    from blacksheep import Application

logger = logging.getLogger(__name__)


class WhatsAppWebhookHandler:
    """
    HTTP webhook handler for WhatsApp messages.

    This class provides webhook endpoints that can be used with
    web frameworks to receive WhatsApp messages.
    """

    def __init__(self, bot: WhatsAppBot):
        """
        Initialize webhook handler.

        Args:
            bot: The WhatsApp bot instance
        """
        self.bot = bot

    async def handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Handle incoming webhook payload.

        Args:
            payload: Webhook payload from WhatsApp

        Returns:
            Response data
        """
        try:
            await self.bot.handle_webhook(payload)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Webhook processing error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def register_with_blacksheep(
        self, app: Application, path: str = "/webhook/whatsapp"
    ) -> None:
        """
        Register webhook handler with BlackSheep application.

        Args:
            app: BlackSheep application instance
            path: Webhook endpoint path
        """
        from blacksheep import Request, Response, json

        @app.router.post(path)
        async def _(request: Request) -> Response:
            try:
                payload = await request.json()
                result = await self.handle_webhook(payload)
                return json(result)
            except Exception as e:
                logger.error(f"Webhook endpoint error: {e}", exc_info=True)
                return json(
                    {"status": "error", "message": "Internal server error"}, status=500
                )
