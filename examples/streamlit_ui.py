"""
Streamlit UI Example

This example demonstrates how to create a chat interface for an agent using Streamlit.
This creates a web application where users can interact with your agent through a chat UI.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.agents.ui.streamlit import AgentToStreamlit

# Create a simple agent
travel_agent = Agent(
    name="Travel Guide",
    description="A helpful travel guide that can answer questions about destinations around the world.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a knowledgeable travel guide who helps users plan trips.
    You provide information about destinations, offer travel tips, suggest itineraries,
    and answer questions about local customs, attractions, and practical travel matters.
    Always be friendly, informative, and considerate of different travel preferences.""",
)

# Convert the agent to a Streamlit app
streamlit_app = AgentToStreamlit(
    title="Travel Assistant",
    description="Ask me anything about travel destinations and planning!",
    initial_mode="presentation",  # Can be "dev" or "presentation"
).adapt(travel_agent)

# This function is what you would put in a Streamlit app.py file
# Run with: streamlit run app.py
if __name__ == "__main__":
    # This will start the Streamlit app
    streamlit_app()
