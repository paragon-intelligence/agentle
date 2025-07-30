===================
Structured Outputs
===================

Agentle allows you to define the structure of agent responses using Pydantic models, enabling strongly-typed outputs that can be easily integrated into your application logic.

Basic Structured Output
---------------------

Here's a simple example of using structured outputs:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Define your output schema
    class WeatherForecast(BaseModel):
        location: str
        current_temperature: float
        conditions: str
        forecast: List[str]
        humidity: Optional[int] = None

    # Create an agent with structured output
    structured_agent = Agent(
        name="Weather Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a weather forecasting assistant. Provide accurate forecasts.",
        response_schema=WeatherForecast  # Define the expected response structure
    )

    # Run the agent
    response = structured_agent.run("What's the weather like in San Francisco?")

    # Access structured data with type hints
    weather = response.parsed
    print(f"Weather for: {weather.location}")
    print(f"Temperature: {weather.current_temperature}°C")
    print(f"Conditions: {weather.conditions}")
    for day in weather.forecast:
        print(f"- {day}")

Working with Complex Models
-------------------------

You can use nested models and complex structures:

.. code-block:: python

    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict
    from enum import Enum
    from datetime import datetime

    # Define an enum for weather conditions
    class WeatherCondition(str, Enum):
        SUNNY = "sunny"
        CLOUDY = "cloudy"
        RAINY = "rainy"
        SNOWY = "snowy"
        STORMY = "stormy"

    # Define a nested model for daily forecast
    class DailyForecast(BaseModel):
        date: datetime
        high_temp: float
        low_temp: float
        condition: WeatherCondition
        precipitation_chance: float = Field(ge=0, le=1)  # Between 0 and 1
        wind_speed: float
        
    # Define the main response model
    class DetailedWeatherForecast(BaseModel):
        location: str
        country: str
        current: Dict[str, float]
        conditions: WeatherCondition
        forecast: List[DailyForecast]
        alerts: Optional[List[str]] = None
        last_updated: datetime

    # Create an agent with the complex schema
    detailed_weather_agent = Agent(
        name="Detailed Weather Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a weather forecasting assistant that provides detailed, structured forecasts.",
        response_schema=DetailedWeatherForecast
    )

    # Get structured response
    response = detailed_weather_agent.run("Give me a detailed 5-day forecast for Tokyo, Japan")
    
    # Access the structured data
    forecast = response.parsed
    print(f"Weather for {forecast.location}, {forecast.country}")
    print(f"Current temperature: {forecast.current['temperature']}°C")
    print(f"Current conditions: {forecast.conditions.value}")
    
    print("\n5-day forecast:")
    for day in forecast.forecast:
        date_str = day.date.strftime("%A, %B %d")
        print(f"{date_str}: {day.condition.value}, {day.low_temp}°C to {day.high_temp}°C")

Combining with Tools
------------------

For even more powerful agents, combine structured outputs with tool calling:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional

    # Define a tool
    def get_city_data(city: str) -> dict:
        """Get basic information about a city."""
        city_database = {
            "Paris": {
                "country": "France",
                "population": 2161000,
                "timezone": "CET",
                "famous_for": ["Eiffel Tower", "Louvre", "Notre Dame"],
            },
            # More cities...
        }
        return city_database.get(city, {"error": f"No data found for {city}"})

    # Define the structured response schema
    class TravelRecommendation(BaseModel):
        city: str
        country: str
        population: int
        local_time: str
        attractions: List[str]
        best_time_to_visit: str
        estimated_daily_budget: float
        safety_rating: Optional[int] = None

    # Create an agent with both tools and a structured output schema
    travel_agent = Agent(
        name="Travel Advisor",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a travel advisor that provides structured recommendations for city visits.""",
        tools=[get_city_data],
        response_schema=TravelRecommendation,
    )

    # Run the agent
    response = travel_agent.run("Create a travel recommendation for Tokyo.")

    # Access structured data
    rec = response.parsed
    print(f"TRAVEL RECOMMENDATION FOR {rec.city}, {rec.country}")
    print(f"Population: {rec.population:,}")
    print(f"Best time to visit: {rec.best_time_to_visit}")

Best Practices
------------

1. **Clear Instructions**: Make sure your agent instructions align with your schema requirements
2. **Schema Complexity**: Balance schema complexity with model capabilities - too complex schemas may lead to validation errors
3. **Field Documentation**: Add field descriptions to help the model generate appropriate values
4. **Optional Fields**: Use Optional for fields that might not always be present
6. **Default Values**: Provide sensible defaults for fields where appropriate