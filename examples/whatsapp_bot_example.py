# examples/whatsapp_bot_example.py
"""
Example of using Agentle agents as WhatsApp bots.
"""

import uvicorn
from blacksheep import Application

from agentle.agents.agent import Agent
from agentle.agents.whatsapp.functions.create_whatsapp_webhook_app import (
    create_whatsapp_webhook_app,
)
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIProvider,
)
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)


def create_webhook_server() -> Application:
    """
    Example 3: Create a web server with webhook endpoint
    """
    # Create agent
    agent = Agent(
        name="WhatsApp Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a helpful assistant. ",
    )

    # Create WhatsApp bot
    evolution_config = EvolutionAPIConfig(
        base_url="http://localhost:8080", instance_name="my-bot", api_key="your-api-key"
    )

    provider = EvolutionAPIProvider(evolution_config)
    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider)

    app = create_whatsapp_webhook_app(whatsapp_bot)

    return app


if __name__ == "__main__":
    # Example 1: Run simple bot
    # asyncio.run(main())

    # Example 3: Run webhook server
    app = create_webhook_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)
