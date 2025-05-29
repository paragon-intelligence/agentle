# examples/whatsapp_bot_example.py
"""
Example of using Agentle agents as WhatsApp bots.
"""

import asyncio
import os
from agentle.agents.agent import Agent
from agentle.agents.whatsapp import (
    AgentToWhatsAppAdapter,
    EvolutionAPIProvider,
    WhatsAppBotConfig,
)
from agentle.agents.whatsapp.providers.evolution import EvolutionAPIConfig
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from blacksheep import Application
import uvicorn


async def main():
    """
    Example 1: Simple WhatsApp bot with Evolution API
    """
    # Create a customer support agent
    agent = Agent(
        name="Customer Support Bot",
        description="A helpful customer support assistant for WhatsApp",
        generation_provider=GoogleGenerationProvider(
            api_key=os.getenv("GOOGLE_API_KEY")
        ),
        model="gemini-2.0-flash",
        instructions="""You are a friendly customer support assistant.
        Keep your responses concise and helpful.
        Use emojis occasionally to be more engaging 😊
        Always be polite and professional.""",
    )

    # Configure Evolution API
    evolution_config = EvolutionAPIConfig(
        base_url=os.getenv("EVOLUTION_API_URL", "http://localhost:8080"),
        instance_name=os.getenv("EVOLUTION_INSTANCE", "customer-support"),
        api_key=os.getenv("EVOLUTION_API_KEY", "your-api-key"),
        webhook_url="https://your-domain.com/webhook/whatsapp",
    )

    # Configure bot behavior
    bot_config = WhatsAppBotConfig(
        typing_indicator=True,
        typing_duration=2,
        auto_read_messages=True,
        welcome_message="👋 Hello! I'm your customer support assistant. How can I help you today?",
        error_message="😔 Sorry, I encountered an error. Please try again or contact human support.",
    )

    # Create WhatsApp provider and adapter
    provider = EvolutionAPIProvider(evolution_config)
    adapter = AgentToWhatsAppAdapter(provider, bot_config)

    # Convert agent to WhatsApp bot
    whatsapp_bot = adapter.adapt(agent)

    # Example 1a: Run bot standalone (for testing)
    print("Starting WhatsApp bot...")
    await whatsapp_bot.start()

    # Keep bot running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping bot...")
        await whatsapp_bot.stop()


async def advanced_example():
    """
    Example 2: Advanced WhatsApp bot with tools and knowledge
    """

    # Define some tools for the agent
    def check_order_status(order_id: str) -> str:
        """Check the status of an order."""
        # In real implementation, this would query a database
        return f"Order {order_id} is currently being processed and will be delivered in 2-3 days."

    def get_product_info(product_name: str) -> str:
        """Get information about a product."""
        products = {
            "laptop": "High-performance laptop with 16GB RAM and 512GB SSD - $999",
            "phone": "Latest smartphone with 5G support and amazing camera - $699",
            "headphones": "Noise-canceling wireless headphones - $299",
        }
        return products.get(product_name.lower(), "Product not found")

    # Create an e-commerce support agent with tools
    agent = Agent(
        name="E-commerce Assistant",
        description="An AI assistant for e-commerce customer support",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are an e-commerce support assistant.
        You can help customers with:
        - Checking order status
        - Product information
        - General inquiries
        
        Be helpful, friendly, and use the available tools when needed.""",
        tools=[check_order_status, get_product_info],
        static_knowledge=[
            "Return Policy: Items can be returned within 30 days of purchase.",
            "Shipping: Free shipping on orders over $50. Standard delivery takes 3-5 business days.",
            "Customer Service Hours: Monday-Friday 9AM-6PM EST",
        ],
    )

    # Create bot with custom configuration
    evolution_config = EvolutionAPIConfig(
        base_url="http://localhost:8080",
        instance_name="ecommerce-support",
        api_key="your-api-key",
    )

    provider = EvolutionAPIProvider(evolution_config)
    adapter = AgentToWhatsAppAdapter(provider)
    whatsapp_bot = adapter.adapt(agent)

    # Add custom webhook handler for special events
    async def handle_custom_events(webhook_payload):
        if webhook_payload.event_type == "connection.update":
            print(f"Connection status: {webhook_payload.data.get('state')}")

    whatsapp_bot.add_webhook_handler(handle_custom_events)

    return whatsapp_bot


def create_webhook_server():
    """
    Example 3: Create a web server with webhook endpoint
    """
    # Create agent
    agent = Agent(
        name="WhatsApp Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a helpful WhatsApp assistant.",
    )

    # Create WhatsApp bot
    evolution_config = EvolutionAPIConfig(
        base_url="http://localhost:8080", instance_name="my-bot", api_key="your-api-key"
    )

    provider = EvolutionAPIProvider(evolution_config)
    adapter = AgentToWhatsAppAdapter(provider)
    whatsapp_bot = adapter.adapt(agent)

    # Create BlackSheep application with webhook
    from agentle.agents.whatsapp.handlers.webhook_handler import (
        create_whatsapp_webhook_app,
    )

    app = create_whatsapp_webhook_app(whatsapp_bot)

    # Add health check endpoint
    @app.router.get("/health")
    async def health_check():
        return {"status": "healthy", "bot": agent.name}

    return app


async def multi_agent_example():
    """
    Example 4: WhatsApp bot with agent team
    """
    from agentle.agents.agent_team import AgentTeam

    # Create specialized agents
    provider = GoogleGenerationProvider()

    sales_agent = Agent(
        name="Sales Agent",
        description="Handles product recommendations and sales inquiries",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You are a sales expert. Help customers find the right products.",
    )

    support_agent = Agent(
        name="Support Agent",
        description="Handles technical support and troubleshooting",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You are a technical support expert. Help customers solve problems.",
    )

    billing_agent = Agent(
        name="Billing Agent",
        description="Handles billing, payments, and refunds",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You are a billing specialist. Help with payment and refund issues.",
    )

    # Create agent team
    team = AgentTeam(
        agents=[sales_agent, support_agent, billing_agent],
        orchestrator_provider=provider,
        orchestrator_model="gemini-2.0-flash",
    )

    # Create WhatsApp bot from team
    evolution_config = EvolutionAPIConfig(
        base_url="http://localhost:8080",
        instance_name="multi-agent-support",
        api_key="your-api-key",
    )

    provider = EvolutionAPIProvider(evolution_config)

    # Note: AgentTeam has the same interface as Agent, so it works with the adapter
    adapter = AgentToWhatsAppAdapter(provider)
    whatsapp_bot = adapter.adapt(team)

    return whatsapp_bot


if __name__ == "__main__":
    # Example 1: Run simple bot
    # asyncio.run(main())

    # Example 3: Run webhook server
    app = create_webhook_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# agentle/agents/whatsapp/providers/whatsapp_business.py
"""
Official WhatsApp Business API implementation (placeholder).
"""

from typing import Optional, Dict, Any
from agentle.agents.whatsapp.providers.base import WhatsAppProvider
from agentle.agents.whatsapp.models import (
    WhatsAppTextMessage,
    WhatsAppMediaMessage,
    WhatsAppContact,
    WhatsAppSession,
    WhatsAppWebhookPayload,
)


class WhatsAppBusinessAPIConfig:
    """Configuration for WhatsApp Business API."""

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        api_version: str = "v17.0",
        webhook_verify_token: str = "",
    ):
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.api_version = api_version
        self.webhook_verify_token = webhook_verify_token


class WhatsAppBusinessAPIProvider(WhatsAppProvider):
    """
    Official WhatsApp Business API implementation.

    This is a placeholder for the official WhatsApp Business API integration.
    The actual implementation would use the Meta/Facebook Graph API.

    Reference: https://developers.facebook.com/docs/whatsapp/cloud-api
    """

    def __init__(self, config: WhatsAppBusinessAPIConfig):
        """
        Initialize WhatsApp Business API provider.

        Args:
            config: WhatsApp Business API configuration
        """
        self.config = config
        # In real implementation:
        # - Initialize HTTP client with proper headers
        # - Set up Graph API endpoints
        # - Configure authentication

    async def initialize(self) -> None:
        """Initialize the WhatsApp Business API connection."""
        # In real implementation:
        # - Verify access token
        # - Check phone number status
        # - Register webhooks if needed
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def shutdown(self) -> None:
        """Shutdown the WhatsApp Business API connection."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def send_text_message(
        self, to: str, text: str, quoted_message_id: Optional[str] = None
    ) -> WhatsAppTextMessage:
        """Send a text message via WhatsApp Business API."""
        # In real implementation:
        # - Format request according to Graph API spec
        # - Send POST to /v17.0/{phone_number_id}/messages
        # - Handle response and create WhatsAppTextMessage
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def send_media_message(
        self,
        to: str,
        media_url: str,
        media_type: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        quoted_message_id: Optional[str] = None,
    ) -> WhatsAppMediaMessage:
        """Send a media message via WhatsApp Business API."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def send_typing_indicator(self, to: str, duration: int = 3) -> None:
        """Send typing indicator via WhatsApp Business API."""
        # Note: WhatsApp Business API uses "read" receipts instead of typing
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def mark_message_as_read(self, message_id: str) -> None:
        """Mark a message as read via WhatsApp Business API."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def get_contact_info(self, phone: str) -> Optional[WhatsAppContact]:
        """Get contact information via WhatsApp Business API."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def get_session(self, phone: str) -> Optional[WhatsAppSession]:
        """Get or create a session for a phone number."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def update_session(self, session: WhatsAppSession) -> None:
        """Update session data."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def process_webhook(self, payload: Dict[str, Any]) -> WhatsAppWebhookPayload:
        """Process incoming webhook data from WhatsApp Business API."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def download_media(self, media_id: str) -> bytes:
        """Download media content by ID via WhatsApp Business API."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this provider."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")

    async def set_webhook_url(self, url: str) -> None:
        """Set the webhook URL for receiving messages."""
        raise NotImplementedError("WhatsApp Business API provider not yet implemented")
