"""
Function Calling Example

This example demonstrates how to create an agent that can call functions/tools
to perform tasks beyond simple text generation.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.cerebras.cerebras_generation_provider import (
    CerebrasGenerationProvider,
)


# Define some simple functions that our agent can use
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        A string describing the weather
    """
    # In a real application, this would call a weather API
    weather_data = {
        "New York": "Sunny, 75°F",
        "London": "Rainy, 60°F",
        "Tokyo": "Cloudy, 65°F",
        "Sydney": "Clear, 80°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculate_mortgage(principal: float, interest_rate: float, years: int) -> str:
    """
    Calculate monthly mortgage payment.

    Args:
        principal: Loan amount in dollars
        interest_rate: Annual interest rate (as a percentage)
        years: Loan term in years

    Returns:
        A string with the monthly payment amount
    """
    monthly_rate = interest_rate / 100 / 12
    num_payments = years * 12
    monthly_payment = (
        principal
        * (monthly_rate * (1 + monthly_rate) ** num_payments)
        / ((1 + monthly_rate) ** num_payments - 1)
    )
    return f"Monthly payment: ${monthly_payment:.2f} for a ${principal} loan at {interest_rate}% over {years} years"


# Create an agent with the tools
agent_with_tools = Agent(
    name="Assistant with Tools",
    generation_provider=CerebrasGenerationProvider(),
    model="llama-3.3-70b",
    instructions="""You are a helpful assistant that can answer questions about the weather 
    and help with financial calculations. Use the provided tools when appropriate.""",
    tools=[get_weather, calculate_mortgage],  # Pass the functions as tools
)

# Run the agent with queries that will likely trigger tool use
agent_response = agent_with_tools.run(
    "What's the weather like in Tokyo? Also, Calculate the monthly payment for a $300,000 mortgage at 4.5% interest for 30 years. Answer in CAPS LOCK."
)
print("Agent response:")
print(agent_response)
print("\n" + "-" * 50 + "\n")
