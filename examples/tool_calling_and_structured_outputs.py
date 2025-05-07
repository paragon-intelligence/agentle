"""
Tool Calling and Structured Outputs Example

This example demonstrates how to create an agent that both calls tools and returns
structured data using a Pydantic model schema.
"""

from pydantic import BaseModel
from typing import Any
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)


# Define a tool the agent can use
def get_city_data(city: str) -> dict[str, Any]:
    """
    Get basic information about a city.

    Args:
        city: The name of the city to get information about

    Returns:
        A dictionary containing city information
    """
    # Mock database of city information
    city_database = {
        "Paris": {
            "country": "France",
            "population": 2161000,
            "timezone": "CET",
            "famous_for": ["Eiffel Tower", "Louvre", "Notre Dame"],
        },
        "New York": {
            "country": "USA",
            "population": 8804190,
            "timezone": "EST",
            "famous_for": ["Times Square", "Statue of Liberty", "Central Park"],
        },
        "Tokyo": {
            "country": "Japan",
            "population": 13960000,
            "timezone": "JST",
            "famous_for": ["Tokyo Tower", "Imperial Palace", "Shibuya Crossing"],
        },
        "Cairo": {
            "country": "Egypt",
            "population": 9540000,
            "timezone": "EET",
            "famous_for": ["Pyramids", "Nile River", "Egyptian Museum"],
        },
    }

    return city_database.get(city, {"error": f"No data found for {city}"})


# Define the structured response schema
class TravelRecommendation(BaseModel):
    city: str
    country: str
    population: int
    local_time: str  # Agent will need to calculate this based on timezone
    attractions: list[str]
    best_time_to_visit: str
    estimated_daily_budget: float
    safety_rating: int | None = None  # 1-10 scale


# Create an agent with both tools and a structured output schema
travel_agent = Agent(
    name="Travel Advisor",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a travel advisor that provides structured recommendations for city visits.
    Use the city data tool to get facts, then enrich the data with your knowledge to create a complete travel guide.
    Always include attractions from the city data when available. Estimate appropriate daily budgets and best times to visit.
    """,
    tools=[get_city_data],
    response_schema=TravelRecommendation,
)

# Run the agent with a query
city_to_analyze = "Tokyo"
response = travel_agent.run(f"Create a travel recommendation for {city_to_analyze}.")

# Check if we got structured data back
rec = response.parsed
print(f"TRAVEL RECOMMENDATION FOR {rec.city.upper()}, {rec.country}")
print(f"Population: {rec.population:,}")
print(f"Local Time: {rec.local_time}")
print("\nTop Attractions:")
for attraction in rec.attractions:
    print(f"- {attraction}")
print(f"\nBest time to visit: {rec.best_time_to_visit}")
print(f"Estimated daily budget: ${rec.estimated_daily_budget:.2f}")
if rec.safety_rating:
    print(f"Safety rating: {rec.safety_rating}/10")
