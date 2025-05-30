from __future__ import annotations

from typing import TYPE_CHECKING

from agentle.agents.whatsapp.webhook.whatsapp_webhook_handler import (
    WhatsAppWebhookHandler,
)
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot

if TYPE_CHECKING:
    from blacksheep import Application


def create_whatsapp_webhook_app(bot: WhatsAppBot) -> Application:
    """
    Create a BlackSheep application with WhatsApp webhook.

    Args:
        bot: WhatsApp bot instance

    Returns:
        BlackSheep application

    Example:
        ```python
        # Create bot
        bot = adapter.adapt(agent)

        # Create web app with webhook
        app = create_whatsapp_webhook_app(bot)

        # Run with uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        ```
    """
    from blacksheep import Application

    app = Application()
    handler = WhatsAppWebhookHandler(bot)
    handler.register_with_blacksheep(app)

    # Start bot when app starts
    @app.on_start
    async def startup():  # type: ignore
        await bot.start()

    # Stop bot when app stops
    @app.on_stop
    async def shutdown():  # type: ignore
        await bot.stop()

    return app
