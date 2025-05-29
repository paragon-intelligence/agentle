from rsb.adapters.adapter import Adapter

from agentle.agents.agent import Agent
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot


class AgentToWhatsAppAdapter(Adapter[Agent, WhatsAppBot]):
    """
    Adapter that converts an Agentle Agent into a WhatsApp bot.

    This adapter provides an easy way to deploy agents as WhatsApp bots,
    handling all the necessary message conversion and session management.
    """

    def __init__(
        self, provider: WhatsAppProvider, config: WhatsAppBotConfig | None = None
    ):
        """
        Initialize the adapter.

        Args:
            provider: WhatsApp provider to use
            config: Bot configuration
        """
        self.provider = provider
        self.config = config or WhatsAppBotConfig()

    def adapt(self, agent: Agent) -> WhatsAppBot:
        """
        Convert an Agent into a WhatsApp bot.

        Args:
            agent: The Agentle agent to convert

        Returns:
            A WhatsApp bot instance

        Example:
            ```python
            from agentle.agents.agent import Agent
            from agentle.agents.whatsapp import (
                AgentToWhatsAppAdapter,
                EvolutionAPIProvider,
                EvolutionAPIConfig
            )

            # Create your agent
            agent = Agent(
                name="Customer Support",
                generation_provider=GoogleGenerationProvider(),
                model="gemini-2.0-flash",
                instructions="You are a helpful customer support agent."
            )

            # Configure Evolution API
            evolution_config = EvolutionAPIConfig(
                base_url="http://localhost:8080",
                instance_name="my-whatsapp-bot",
                api_key="your-api-key"
            )

            # Create provider and adapter
            provider = EvolutionAPIProvider(evolution_config)
            adapter = AgentToWhatsAppAdapter(provider)

            # Convert agent to WhatsApp bot
            whatsapp_bot = adapter.adapt(agent)

            # Start the bot
            await whatsapp_bot.start()
            ```
        """
        return WhatsAppBot(agent=agent, provider=self.provider, config=self.config)
