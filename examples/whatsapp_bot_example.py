# examples/whatsapp_bot_example.py
"""
Example of using Agentle agents as WhatsApp bots with session management.
"""

import asyncio
import os
import uvicorn
from blacksheep import Application
from typing import Any

from agentle.agents.agent import Agent
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
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
from agentle.session.session_manager import SessionManager
from agentle.session.in_memory_session_store import InMemorySessionStore

# Optional: Use Redis for production
try:
    from agentle.session.redis_session_store import RedisSessionStore

    redis_available = True
except ImportError:
    redis_available = False
    RedisSessionStore = None


def get_weather(location: str) -> str:
    """Example tool to get weather information."""
    # This is a mock implementation
    weather_data = {
        "São Paulo": "Sunny, 25°C",
        "Rio de Janeiro": "Partly cloudy, 28°C",
        "New York": "Rainy, 15°C",
        "London": "Foggy, 12°C",
        "Tokyo": "Clear, 20°C",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


async def main():
    """
    Example 1: Simple bot with in-memory sessions
    """
    print("Starting WhatsApp bot with in-memory sessions...")

    # Create agent with tools
    agent = Agent(
        name="Weather Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a helpful weather assistant that can provide weather information.
        Always be friendly and helpful. If someone asks about weather, use the get_weather tool.""",
        tools=[get_weather],
    )

    # Configure Evolution API
    evolution_config = EvolutionAPIConfig(
        base_url=os.getenv("EVOLUTION_API_URL", "http://localhost:8080"),
        instance_name=os.getenv("EVOLUTION_INSTANCE_NAME", "my-bot"),
        api_key=os.getenv("EVOLUTION_API_KEY", "your-api-key"),
    )

    # Create session manager with in-memory storage
    from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession

    session_store = InMemorySessionStore[WhatsAppSession]()
    session_manager = SessionManager[WhatsAppSession](
        session_store=session_store,
        default_ttl_seconds=1800,  # 30 minutes
        enable_events=True,
    )

    # Add event handlers for session lifecycle
    async def on_session_created(session_id: str, session_data: Any) -> None:
        print(f"📱 New session created: {session_id}")

    async def on_session_deleted(session_id: str, session_data: Any) -> None:
        print(f"🗑️ Session deleted: {session_id}")

    session_manager.add_event_handler("session_created", on_session_created)
    session_manager.add_event_handler("session_deleted", on_session_deleted)

    # Create provider with session management
    provider = EvolutionAPIProvider(
        config=evolution_config,
        session_manager=session_manager,
        session_ttl_seconds=1800,
    )

    # Configure bot behavior
    bot_config = WhatsAppBotConfig(
        typing_indicator=True,
        typing_duration=2,
        auto_read_messages=True,
        session_timeout_minutes=30,
        welcome_message="Hello! I'm your weather assistant. Ask me about the weather in any city!",
        error_message="Sorry, I encountered an error. Please try again in a moment.",
    )

    # Create WhatsApp bot
    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)

    try:
        # Initialize the bot
        await whatsapp_bot.start()

        # Set webhook URL if provided
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            await provider.set_webhook_url(webhook_url)
            print(f"Webhook URL set to: {webhook_url}")

        print("Bot is running! Press Ctrl+C to stop.")

        # Keep the bot running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Stopping bot...")
    finally:
        await whatsapp_bot.stop()


def create_production_webhook_server() -> Application:
    """
    Example 2: Production webhook server with Redis sessions
    """
    print("Creating production webhook server...")

    # Create agent
    agent = Agent(
        name="Production WhatsApp Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a professional customer service assistant. 
        Be helpful, polite, and provide accurate information. You can help with weather inquiries.""",
        tools=[get_weather],
    )

    # Configure Evolution API
    evolution_config = EvolutionAPIConfig(
        base_url=os.getenv("EVOLUTION_API_URL", "http://localhost:8080"),
        instance_name=os.getenv("EVOLUTION_INSTANCE_NAME", "production-bot"),
        api_key=os.getenv("EVOLUTION_API_KEY", "your-api-key"),
    )

    # Choose session store based on environment
    from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession

    if redis_available and os.getenv("REDIS_URL") and RedisSessionStore is not None:
        print("Using Redis session store for production")
        session_store = RedisSessionStore[WhatsAppSession](
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            key_prefix="whatsapp:sessions:",
            default_ttl_seconds=3600,  # 1 hour
            session_class=WhatsAppSession,
        )
    else:
        print("Using in-memory session store (development)")
        session_store = InMemorySessionStore[WhatsAppSession]()

    # Create session manager
    session_manager = SessionManager[WhatsAppSession](
        session_store=session_store, default_ttl_seconds=3600, enable_events=True
    )

    # Create provider
    provider = EvolutionAPIProvider(
        config=evolution_config,
        session_manager=session_manager,
        session_ttl_seconds=3600,
    )

    # Configure bot for production
    bot_config = WhatsAppBotConfig(
        typing_indicator=True,
        typing_duration=3,
        auto_read_messages=True,
        session_timeout_minutes=60,
        max_message_length=4000,
        welcome_message="Welcome! I'm here to help you. How can I assist you today?",
        error_message="I apologize for the inconvenience. Please try again later or contact support.",
    )

    # Create WhatsApp bot
    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)

    # Convert to BlackSheep application
    return whatsapp_bot.to_blacksheep_app(
        webhook_path="/webhook/whatsapp",
        show_error_details=os.getenv("DEBUG", "false").lower() == "true",
    )


def create_development_server() -> Application:
    """
    Example 3: Development server with detailed logging
    """
    print("Creating development server...")

    # Enable debug logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Create a simple agent
    agent = Agent(
        name="Dev WhatsApp Bot",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a development bot for testing WhatsApp integration.",
        tools=[get_weather],
        debug=True,  # Enable agent debugging
    )

    # Configure Evolution API for development
    evolution_config = EvolutionAPIConfig(
        base_url="http://localhost:8080",
        instance_name="dev-bot",
        api_key="dev-api-key",
        timeout=60,  # Longer timeout for debugging
    )

    # Use in-memory sessions for development
    from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession

    session_store = InMemorySessionStore[WhatsAppSession](cleanup_interval_seconds=60)
    session_manager = SessionManager[WhatsAppSession](
        session_store=session_store,
        default_ttl_seconds=600,  # 10 minutes for quick testing
        enable_events=True,
    )

    # Create provider
    provider = EvolutionAPIProvider(
        config=evolution_config,
        session_manager=session_manager,
        session_ttl_seconds=600,
    )

    # Configure bot for development
    bot_config = WhatsAppBotConfig(
        typing_indicator=True,
        typing_duration=1,  # Faster for development
        auto_read_messages=True,
        session_timeout_minutes=10,
        welcome_message="🔧 Development Bot - Ready for testing!",
        error_message="💥 Error in development bot - check logs!",
    )

    # Create bot
    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)

    return whatsapp_bot.to_blacksheep_app(
        webhook_path="/webhook/whatsapp",
        show_error_details=True,  # Show detailed errors in development
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = os.getenv("BOT_MODE", "webhook").lower()

    if mode == "simple":
        # Example 1: Simple bot (async)
        asyncio.run(main())
    elif mode == "production":
        # Example 2: Production webhook server
        app = create_production_webhook_server()
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif mode == "dev" or mode == "development":
        # Example 3: Development server
        app = create_development_server()
        uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
    else:
        # Default: webhook mode
        app = create_production_webhook_server()
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
