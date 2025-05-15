"""
Structured Outputs Example

This example demonstrates how to create an agent that returns structured data
using a Pydantic model as a response schema.
"""

from typing import Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from agentle.agents.agent import Agent
from agentle.generations.providers.google import GoogleGenerationProvider
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

load_dotenv()

observability_client = LangfuseObservabilityClient()


# Define a structured response schema using Pydantic
class WeatherForecast(BaseModel):
    location: str
    current_temperature: float
    conditions: str
    forecast: list[str]
    humidity: Optional[int] = None


# Create an agent with the response schema
structured_agent = Agent(
    name="Weather Agent",
    generation_provider=GoogleGenerationProvider(
        use_vertex_ai=True,
        tracing_client=observability_client,
        project=os.getenv("GOOGLE_PROJECT_ID"),
        location=os.getenv("GOOGLE_LOCATION"),
    ),
    model="gemini-2.0-flash",
    instructions="You are a weather forecasting assistant. When asked about weather, provide accurate forecasts.",
    response_schema=WeatherForecast,  # This defines the expected response structure
)

# Run the agent with a query that requires structured data
response = structured_agent.run("What's the weather like in San Francisco?")


weather = response.parsed
print(f"Weather for: {weather.location}")
print(f"Temperature: {weather.current_temperature}°C")
print(f"Conditions: {weather.conditions}")
print("Forecast:")
for day in weather.forecast:
    print(f"- {day}")
if weather.humidity is not None:
    print(f"Humidity: {weather.humidity}%")
