from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)


agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    static_knowledge=["examples/arthur.md"],
    instructions="""Você é uma assistente de IA responsável por responder perguntas e conversar, de maneira educada, sobre o Arthur.""",
)

print(agent.run("Boa noite. quem é o arthur").pretty_formatted())
