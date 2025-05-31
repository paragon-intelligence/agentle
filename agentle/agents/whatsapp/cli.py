# agentle/agents/whatsapp/cli.py
"""
CLI tool for quickly creating WhatsApp bots from Agentle agents.
"""

import asyncio
from collections.abc import MutableSequence
import os
from typing import Optional, TextIO

import click

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
from agentle.generations.providers.openai.openai import OpenaiGenerationProvider
from agentle.generations.tools.tool import Tool


@click.group()
def cli():
    """Agentle WhatsApp Bot CLI"""
    pass


@cli.command()
@click.option("--name", default="WhatsApp Assistant", help="Bot name")
@click.option(
    "--provider",
    type=click.Choice(["google", "openai"]),
    default="google",
    help="AI provider",
)
@click.option("--model", help="Model to use (e.g., gemini-2.0-flash, gpt-4)")
@click.option(
    "--evolution-url", default="http://localhost:8080", help="Evolution API URL"
)
@click.option("--instance", required=True, help="Evolution instance name")
@click.option("--api-key", required=True, help="Evolution API key")
@click.option("--ai-key", envvar="AI_API_KEY", help="AI provider API key")
@click.option("--instructions", help="Custom instructions for the bot")
@click.option("--welcome", help="Welcome message for new conversations")
def run(
    name: str,
    provider: str,
    model: Optional[str],
    evolution_url: str,
    instance: str,
    api_key: str,
    ai_key: Optional[str],
    instructions: Optional[str],
    welcome: Optional[str],
):
    """Run a WhatsApp bot"""

    # Create AI provider
    if provider == "google":
        if not ai_key:
            ai_key = os.getenv("GOOGLE_API_KEY")
        if not ai_key:
            raise click.ClickException(
                "Google API key required (--ai-key or GOOGLE_API_KEY env)"
            )
        generation_provider = GoogleGenerationProvider(api_key=ai_key)
        default_model = "gemini-2.0-flash"
    else:  # openai
        if not ai_key:
            ai_key = os.getenv("OPENAI_API_KEY")
        if not ai_key:
            raise click.ClickException(
                "OpenAI API key required (--ai-key or OPENAI_API_KEY env)"
            )
        generation_provider = OpenaiGenerationProvider(api_key=ai_key)
        default_model = "gpt-4"

    # Use provided model or default
    model = model or default_model

    # Create agent
    agent = Agent(
        name=name,
        generation_provider=generation_provider,
        model=model,
        instructions=instructions
        or f"You are {name}, a helpful WhatsApp assistant. Be concise and friendly.",
    )

    # Configure Evolution API
    evolution_config = EvolutionAPIConfig(
        base_url=evolution_url, instance_name=instance, api_key=api_key
    )

    # Configure bot
    bot_config = WhatsAppBotConfig(
        welcome_message=welcome, typing_indicator=True, auto_read_messages=True
    )

    # Create and run bot
    evolution_provider = EvolutionAPIProvider(evolution_config)
    whatsapp_bot = WhatsAppBot(
        agent=agent, provider=evolution_provider, config=bot_config
    )

    click.echo(f"🚀 Starting {name} on Evolution instance '{instance}'...")
    click.echo(f"📱 Using {provider} provider with model {model}")
    click.echo("Press Ctrl+C to stop")

    async def run_bot():
        await whatsapp_bot.start()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            click.echo("\n🛑 Stopping bot...")
            await whatsapp_bot.stop()

    asyncio.run(run_bot())


@cli.command()
@click.option("--port", default=8000, help="Port for webhook server")
@click.option("--host", default="0.0.0.0", help="Host for webhook server")
@click.option("--name", default="WhatsApp Webhook Server", help="Server name")
@click.option(
    "--evolution-url", default="http://localhost:8080", help="Evolution API URL"
)
@click.option("--instance", required=True, help="Evolution instance name")
@click.option("--api-key", required=True, help="Evolution API key")
@click.option(
    "--webhook-path", default="/webhook/whatsapp", help="Webhook endpoint path"
)
def serve(
    port: int,
    host: str,
    name: str,
    evolution_url: str,
    instance: str,
    api_key: str,
    webhook_path: str,
):
    """Run a webhook server for WhatsApp bot"""
    import uvicorn

    # Create a simple agent
    agent = Agent(
        name=name,
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a helpful WhatsApp assistant.",
    )

    # Create WhatsApp bot
    evolution_config = EvolutionAPIConfig(
        base_url=evolution_url, instance_name=instance, api_key=api_key
    )

    evolution_provider = EvolutionAPIProvider(evolution_config)
    whatsapp_bot = WhatsAppBot(agent=agent, provider=evolution_provider)

    # Create web app
    app = whatsapp_bot.to_blacksheep_app()

    # Health check
    @app.router.get("/health")
    async def health():  # type: ignore
        return {"status": "healthy", "bot": name}

    # Startup/shutdown
    @app.on_start
    async def startup():  # type: ignore
        await whatsapp_bot.start()
        # Set webhook URL
        public_url = os.getenv("PUBLIC_URL", f"http://localhost:{port}")
        webhook_url = f"{public_url}{webhook_path}"
        await evolution_provider.set_webhook_url(webhook_url)
        click.echo(f"📍 Webhook URL: {webhook_url}")

    @app.on_stop
    async def shutdown():  # type: ignore
        await whatsapp_bot.stop()

    click.echo(f"🌐 Starting webhook server on {host}:{port}")
    click.echo(f"📱 Evolution instance: {instance}")
    click.echo(f"🔗 Webhook path: {webhook_path}")

    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("config_file", type=click.File("r"))
def from_config(config_file: TextIO):
    """Run WhatsApp bot from configuration file"""
    import yaml

    config = yaml.safe_load(config_file)

    # Extract configuration
    bot_config = config.get("bot", {})
    evolution_config = config.get("evolution", {})
    agent_config = config.get("agent", {})

    # Create agent from config
    provider_name = agent_config.get("provider", "google")
    if provider_name == "google":
        generation_provider = GoogleGenerationProvider(
            api_key=agent_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        )
    else:
        _openai_api_key = agent_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if _openai_api_key is None:
            raise ValueError("OpenAI API key is required")
        generation_provider = OpenaiGenerationProvider(api_key=_openai_api_key)

    # Load tools if specified
    tools: MutableSequence[Tool] = []
    for tool_config in agent_config.get("tools", []):  # type: ignore TODO
        # In real implementation, dynamically load tools
        pass

    agent = Agent(
        name=agent_config.get("name", "WhatsApp Bot"),
        generation_provider=generation_provider,
        model=agent_config.get("model"),
        instructions=agent_config.get("instructions", ""),
        tools=tools,
        static_knowledge=agent_config.get("knowledge", []),
    )

    # Create Evolution provider
    evolution_provider = EvolutionAPIProvider(
        EvolutionAPIConfig(
            base_url=evolution_config.get("url", "http://localhost:8080"),
            instance_name=evolution_config["instance"],
            api_key=evolution_config["api_key"],
        )
    )

    # Create bot
    bot_settings = WhatsAppBotConfig(
        welcome_message=bot_config.get("welcome_message"),
        error_message=bot_config.get("error_message"),
        typing_indicator=bot_config.get("typing_indicator", True),
        auto_read_messages=bot_config.get("auto_read_messages", True),
    )

    whatsapp_bot = WhatsAppBot(
        agent=agent, provider=evolution_provider, config=bot_settings
    )

    click.echo(f"🚀 Starting bot from config: {config_file.name}")

    async def run_bot():
        await whatsapp_bot.start()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await whatsapp_bot.stop()

    asyncio.run(run_bot())


if __name__ == "__main__":
    cli()


# Example configuration file: whatsapp_bot_config.yaml
"""
bot:
  welcome_message: "👋 Hello! I'm your AI assistant. How can I help you today?"
  error_message: "😔 Sorry, something went wrong. Please try again."
  typing_indicator: true
  auto_read_messages: true

evolution:
  url: "http://localhost:8080"
  instance: "my-whatsapp-bot"
  api_key: "your-evolution-api-key"

agent:
  name: "Customer Support Bot"
  provider: "google"
  model: "gemini-2.0-flash"
  api_key: null  # Uses env variable
  instructions: |
    You are a friendly customer support assistant.
    Be helpful, concise, and professional.
    Use emojis occasionally to be more engaging.
  knowledge:
    - "Return policy: Items can be returned within 30 days."
    - "Shipping: Free shipping on orders over $50."
    - "Hours: Monday-Friday 9AM-6PM EST"
  tools: []  # Tool configuration would go here
"""
