"""
Structured Outputs Example

This example demonstrates how to create an agent that returns structured data
using a Pydantic model as a response schema.
"""

from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.trace_params import TraceParams
from agentle.generations.providers.cerebras.cerebras_generation_provider import (
    CerebrasGenerationProvider,
)
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

load_dotenv()

tracing_client = LangfuseObservabilityClient()


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
    generation_provider=CerebrasGenerationProvider(),
    model="category_standard_experimental",
    instructions="You are a weather forecasting assistant. When asked about weather, provide accurate forecasts.",
    response_schema=WeatherForecast,  # This defines the expected response structure
    config=AgentConfig(
        generationConfig=GenerationConfig(
            trace_params=TraceParams(
                name="Weather App",
                user_id="test_user",
            )
        )
    ),
)

# Run the agent with a query that requires structured data
response = structured_agent.run("Qual vai ser o clima em uberaba hoje?")


weather = response.parsed
print(f"Weather for: {weather.location}")
print(f"Temperature: {weather.current_temperature}°C")
print(f"Conditions: {weather.conditions}")
print("Forecast:")
for day in weather.forecast:
    print(f"- {day}")
if weather.humidity is not None:
    print(f"Humidity: {weather.humidity}%")
