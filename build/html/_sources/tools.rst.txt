=====
Tools
=====

Tools (also known as function calling) allow agents to perform actions beyond text generation by invoking specific functions. This page explains how to integrate and use tools with Agentle agents.

Basic Tool Integration
--------------------

Here's a simple example of integrating a tool with an agent:

.. code-block:: python

    def get_weather(location: str) -> str:
        """
        Get the current weather for a location.

        Args:
            location: The city or location to get weather for

        Returns:
            A string describing the weather
        """
        weather_data = {
            "New York": "Sunny, 75°F",
            "London": "Rainy, 60°F",
            "Tokyo": "Cloudy, 65°F",
            "Sydney": "Clear, 80°F",
        }
        return weather_data.get(location, f"Weather data not available for {location}")

    # Create an agent with a tool
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    weather_agent = Agent(
        name="Weather Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant that can answer questions about the weather.",
        tools=[get_weather]  # Pass the function as a tool
    )

    # The agent will automatically use the tool when appropriate
    response = weather_agent.run("What's the weather like in Tokyo?")
    print(response.text)

Function Docstrings and Type Hints
---------------------------------

Agentle uses function docstrings and type hints to determine:

1. When to call the function
2. What parameters to pass
3. How to interpret the results

For best results, provide clear docstrings and type hints:

.. code-block:: python

    def calculate_mortgage(
        principal: float,
        interest_rate: float,
        years: int
    ) -> dict:
        """
        Calculate monthly mortgage payments.

        Args:
            principal: The loan amount in dollars
            interest_rate: Annual interest rate (as a percentage, e.g., 5.5 for 5.5%)
            years: Loan term in years

        Returns:
            A dictionary containing monthly payment, total interest, and total cost
        """
        monthly_rate = interest_rate / 100 / 12
        num_payments = years * 12
        
        # Calculate monthly payment
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        
        # Calculate total interest and total cost
        total_cost = monthly_payment * num_payments
        total_interest = total_cost - principal
        
        return {
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_cost": round(total_cost, 2)
        }

Adding Multiple Tools
-------------------

You can add multiple tools to an agent:

.. code-block:: python

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        # Implementation...
        return weather_data.get(location, f"Weather data not available for {location}")
    
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert an amount from one currency to another.
        
        Args:
            amount: The amount to convert
            from_currency: The source currency code (e.g., USD, EUR)
            to_currency: The target currency code (e.g., USD, EUR)
            
        Returns:
            The converted amount
        """
        # Sample conversion rates (in practice, use a real API)
        rates = {
            "USD": 1.0,
            "EUR": 0.85,
            "GBP": 0.75,
            "JPY": 110.0,
            "CAD": 1.25
        }
        
        # Convert to USD first, then to target currency
        usd_amount = amount / rates.get(from_currency, 1.0)
        converted_amount = usd_amount * rates.get(to_currency, 1.0)
        
        return round(converted_amount, 2)
    
    # Create an agent with multiple tools
    travel_assistant = Agent(
        name="Travel Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful travel assistant.",
        tools=[get_weather, convert_currency]  # Multiple tools
    )

Advanced Tool Usage
-----------------

Combining Tools with Structured Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For even more powerful agents, combine tool calling with structured outputs:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional

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

Classes as Tools
~~~~~~~~~~~~~~

You can also use methods from classes as tools:

.. code-block:: python

    class Calculator:
        def add(self, a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b
    
        def subtract(self, a: float, b: float) -> float:
            """Subtract b from a."""
            return a - b
    
    calculator = Calculator()
    
    # Use instance methods as tools
    math_agent = Agent(
        name="Math Helper",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a math assistant.",
        tools=[calculator.add, calculator.subtract]
    )

Best Practices for Tools
----------------------

1. **Clear Docstrings**: Provide clear, detailed docstrings that explain what the function does
2. **Type Hints**: Always use type hints for parameters and return values
3. **Error Handling**: Ensure your tools handle errors gracefully
4. **Idempotence**: When possible, make your tools idempotent (same input always produces same output)
5. **Security**: Be mindful of security implications, especially for tools that access sensitive resources