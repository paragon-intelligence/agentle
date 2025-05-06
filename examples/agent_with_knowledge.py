from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)


agent = Agent(
    name="Research Assistant",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You help analysing websites.",
    # Array of string-based knowledge sources (no caching)
    static_knowledge=[
        # URLs as strings
        "https://monowave.store/",
    ],
)

agent.run("Que tipos de produtos tem a monowave?")
