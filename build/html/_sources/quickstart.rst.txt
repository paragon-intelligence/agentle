==========
Quickstart
==========

This guide will help you get started with Agentle by creating your first agent and exploring basic functionality.

Creating Your First Agent
-----------------------

Let's start by creating a simple agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
    import os

    # Create a generation provider (Google AI in this example)
    provider = GoogleGenerationProvider(
        api_key=os.environ.get("GOOGLE_API_KEY")  # Get API key from environment variable
    )

    # Create a simple agent
    agent = Agent(
        name="Quick Start Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant who provides concise, accurate information."
    )

    # Run the agent
    response = agent.run("What are the three laws of robotics?")

    # Print the response
    print(response.text)

Save this code to a file (e.g., ``quickstart.py``) and run it:

.. code-block:: bash

    python quickstart.py

Adding Tools
----------

Now let's enhance our agent by adding a tool:

.. code-block:: python

    def get_weather(location: str) -> str:
        """
        Get the current weather for a location.

        Args:
            location: The city or location to get weather for

        Returns:
            A string describing the weather
        """
        # In a real application, you would call a weather API here
        weather_data = {
            "New York": "Sunny, 75°F",
            "London": "Rainy, 60°F",
            "Tokyo": "Cloudy, 65°F",
            "Paris": "Partly cloudy, 70°F",
            "Sydney": "Clear, 80°F",
        }
        return weather_data.get(location, f"Weather data not available for {location}")

    # Create an agent with a tool
    weather_agent = Agent(
        name="Weather Assistant",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant that can answer questions about the weather.",
        tools=[get_weather]  # Pass the function as a tool
    )

    # The agent will automatically use the tool when appropriate
    response = weather_agent.run("What's the weather like in Tokyo?")
    print(response.text)

    # Ask a question that should trigger tool use
    response = weather_agent.run("Can you tell me the weather in London and Paris?")
    print(response.text)

Creating Structured Outputs
-------------------------

Let's create an agent that returns strongly-typed results using Pydantic:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional

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
        generation_provider=provider,
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
    print("Forecast:")
    for day in weather.forecast:
        print(f"- {day}")
    if weather.humidity is not None:
        print(f"Humidity: {weather.humidity}%")

Adding Static Knowledge
--------------------

Let's create an agent with domain-specific knowledge:

.. code-block:: python

    from agentle.agents.knowledge.static_knowledge import StaticKnowledge

    # Create an agent with static knowledge
    travel_expert = Agent(
        name="Japan Travel Expert",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a Japan travel expert who provides detailed information about Japanese destinations.",
        # Provide static knowledge from multiple sources
        static_knowledge=[
            # Include knowledge from a local document (if you have this file)
            # StaticKnowledge(content="data/japan_travel_guide.pdf", cache=3600),
            
            # Include direct text knowledge
            "Tokyo is the capital of Japan and one of the most populous cities in the world.",
            "Cherry blossom season in Japan typically runs from late March to early April.",
            "Mount Fuji is Japan's tallest mountain at 3,776 meters and is considered one of Japan's three sacred mountains.",
            "Kyoto was the imperial capital of Japan for more than 1,000 years and is famous for its temples, shrines, and traditional wooden houses."
        ]
    )

    # The agent will incorporate the knowledge when answering
    response = travel_expert.run("What should I know about visiting Tokyo in cherry blossom season?")
    print(response.text)

Creating a Multi-Agent Pipeline
----------------------------

Now let's create a pipeline of specialized agents:

.. code-block:: python

    from agentle.agents.agent_pipeline import AgentPipeline

    # Create specialized agents
    research_agent = Agent(
        name="Research Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a research agent focused on gathering information.
        Be thorough and prioritize accuracy over speculation."""
    )

    analysis_agent = Agent(
        name="Analysis Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are an analysis agent that identifies patterns.
        Highlight meaningful relationships and insights from the data."""
    )

    summary_agent = Agent(
        name="Summary Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a summary agent that creates concise summaries.
        Present key findings in a logical order with accessible language."""
    )

    # Create a pipeline
    pipeline = AgentPipeline(
        agents=[research_agent, analysis_agent, summary_agent],
        debug_mode=True  # Enable to see intermediate steps
    )

    # Run the pipeline
    result = pipeline.run("Research the impact of artificial intelligence on healthcare")
    print("\nFinal Result:")
    print(result.text)

Creating a Web API
---------------

Let's deploy an agent as a web API:

.. code-block:: python

    from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
    import uvicorn

    # Create your agent
    code_assistant = Agent(
        name="Code Assistant",
        description="An AI assistant specialized in helping with programming tasks.",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a helpful programming assistant.
        You can answer questions about programming languages, help debug code,
        explain programming concepts, and provide code examples.""",
    )

    # Convert the agent to a BlackSheep ASGI application
    app = AgentToBlackSheepApplicationAdapter().adapt(code_assistant)

    # Run the API server
    if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=8000)

To test this API:

.. code-block:: bash

    # In a new terminal
    curl -X POST "http://localhost:8000/api/v1/agents/code_assistant/run" \
        -H "Content-Type: application/json" \
        -d '{"input": "Write a Python function to calculate the Fibonacci sequence"}'

Creating a Streamlit Interface
---------------------------

Let's create a chat interface for our agent:

.. code-block:: python

    from agentle.agents.ui.streamlit import AgentToStreamlit

    # Create your agent
    travel_agent = Agent(
        name="Travel Guide",
        description="A helpful travel guide that answers questions about destinations.",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a knowledgeable travel guide who helps users plan trips.""",
    )

    # Convert the agent to a Streamlit app
    streamlit_app = AgentToStreamlit(
        title="Travel Assistant",
        description="Ask me anything about travel destinations and planning!",
        initial_mode="presentation",  # Can be "dev" or "presentation"
    ).adapt(travel_agent)

    # Run the Streamlit app
    if __name__ == "__main__":
        streamlit_app()

Save this as `streamlit_app.py` and run:

.. code-block:: bash

    streamlit run streamlit_app.py

Next Steps
---------

Now that you've created your first agents, you can:

1. Learn more about :doc:`agents` and their capabilities
2. Explore :doc:`tools` for extending agent functionality
3. Discover how to use :doc:`structured_outputs` for type-safe responses
4. Learn about :doc:`agent_composition` for creating more complex systems
5. See how to enhance agents with :doc:`knowledge_integration`

For a comprehensive overview of all features, check out the documentation sections in the sidebar.