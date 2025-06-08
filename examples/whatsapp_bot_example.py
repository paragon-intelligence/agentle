# examples/whatsapp_bot_example.py
"""
Example of using Agentle agents as WhatsApp bots with session management.
"""

import os

import uvicorn
from blacksheep import Application

from agentle.agents.agent import Agent
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
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
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager
from dotenv import load_dotenv

load_dotenv()

def create_server() -> Application:
    """
    Example 2: Production webhook server with Redis sessions
    """
    agent = Agent(
        name="Production WhatsApp Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        static_knowledge=["examples/arthur.md"],
        instructions="""Você é uma assistente de IA responsável por responder perguntas e conversar, de maneira educada, sobre o Arthur. Em sua resposta final, utilize algum tipo de formato que o WhatApp entenda. Acredito que markdown funciona, mas garanta que funciona. Suas mensagens devem ser bem curtas e diretas, mas nao tao curtas. Respondendo apenas o necessario. Converse com um sotaque mineiro. Fingindo que é da roça. Use emojis do agro.""",
    )

    session_manager = SessionManager[WhatsAppSession](
        session_store=InMemorySessionStore[WhatsAppSession](),
        default_ttl_seconds=3600,
        enable_events=True,
    )

    # Create provider
    provider = EvolutionAPIProvider(
        config=EvolutionAPIConfig(
            base_url=os.getenv("EVOLUTION_API_URL", "http://localhost:8080"),
            instance_name=os.getenv("EVOLUTION_INSTANCE_NAME", "production-bot"),
            api_key=os.getenv("EVOLUTION_API_KEY", "your-api-key"),
        ),
        session_manager=session_manager,
        session_ttl_seconds=3600,
    )

    # Configure bot for production
    bot_config = WhatsAppBotConfig(
        typing_indicator=True,
        typing_duration=8,
        auto_read_messages=True,
        session_timeout_minutes=60,
        max_message_length=4000,
        welcome_message="""Olá! Seja muito bem-vindo(a)!  
Sou a assistente pessoal do Arthur 🤖  
Já estou te redirecionando para uma de nossas assistentes especializadas, que vai te ajudar melhor 💬

Aqui, você pode:

📌 Saber mais sobre **quem é o Arthur**  
🎯 Descobrir mais sobre a **carreira do Arthur**

Eu fui configurada especificamente com uma persona do agro, que envia muitos emojis e escreve como se estivesse falando em mineirês! Isso é apenas um teste. Posso trocar a persona da IA para ela atuar como eu quiser. Atualmente, não suporto imagens nem mensagens de voz, mas isso está em desenvolvimento.

Fique à vontade! Estamos aqui para te ajudar!
        """,
        error_message="Desculpe pelo inconveniente. Por favor, tente novamente mais tarde ou contate o suporte.",
    )

    # Create WhatsApp bot
    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)

    # Convert to BlackSheep application
    return whatsapp_bot.to_blacksheep_app(
        webhook_path="/webhook/whatsapp",
        show_error_details=os.getenv("DEBUG", "false").lower() == "true",
    )


app = create_server()
port = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
