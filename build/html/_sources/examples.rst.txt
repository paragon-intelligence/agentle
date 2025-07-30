Examples
========

This page provides a variety of examples to help you understand how to use Agentle.

Quick Start
----------

Here's a simple example to get you started with Agentle:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a simple agent
    agent = Agent(
        name="Quick Start Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant who provides concise, accurate information."
    )

    # Run the agent
    response = agent.run("What are the three laws of robotics?")

    # Print the response
    print(response.text)

Agent Creation
-------------

Creating agents with different generation providers:

.. code-block:: python

    # With Google Gemini
    google_agent = Agent(
        name="Google Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant powered by Google Gemini."
    )

    # With OpenAI
    openai_agent = Agent(
        name="OpenAI Agent",
        generation_provider=OpenAIGenerationProvider(),
        model="gpt-4o",
        instructions="You are a helpful assistant powered by OpenAI GPT-4."
    )

Static Knowledge Integration
---------------------------

Agents can be initialized with static knowledge from various sources:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
    from agentle.knowledge.knowledge_source import FileKnowledgeSource, TextKnowledgeSource, URLKnowledgeSource

    # Create knowledge sources
    file_source = FileKnowledgeSource(file_path="data/product_catalog.txt")
    text_source = TextKnowledgeSource(text="The company was founded in 2023 and specializes in AI agent frameworks.")
    url_source = URLKnowledgeSource(url="https://example.com/about")

    # Create agent with knowledge
    agent = Agent(
        name="Product Support Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a product support agent. Answer questions based on the provided knowledge.",
        knowledge_sources=[file_source, text_source, url_source]
    )

    # The agent will use this knowledge when answering questions
    response = agent.run("Tell me about your product catalog.")
    print(response.text)

Tool Integration
--------------

Integrate external tools and functions with your agents:

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

Structured Outputs
----------------

Get structured, type-safe responses from your agents:

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

Agent Pipelines
-------------

Chain multiple specialized agents together in a sequential pipeline:

.. code-block:: python

    from agentle.agents.agent import Agent
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
    print(result.text)

Agent Teams
---------

Build collaborative agent teams that can work together on complex tasks:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.agent_team import AgentTeam
    
    # Create specialized team members
    researcher = Agent(
        name="Researcher",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a researcher who finds factual information about topics."
    )
    
    creative_writer = Agent(
        name="Creative Writer",
        generation_provider=provider,
        model="gemini-2.5-flash", 
        instructions="You are a creative writer who can produce engaging content."
    )
    
    editor = Agent(
        name="Editor",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are an editor who improves and refines content."
    )
    
    # Create a team with a coordinator
    content_team = AgentTeam(
        name="Content Creation Team",
        agents=[researcher, creative_writer, editor],
        coordinator_instructions="""
        You are coordinating a team to create content. 
        First, have the Researcher gather facts about the topic.
        Then, ask the Creative Writer to create engaging content using those facts.
        Finally, have the Editor refine and improve the final content.
        """
    )
    
    # Run the team on a task
    result = content_team.run("Create a blog post about sustainable energy solutions")
    print(result.text)

Web API with BlackSheep
---------------------

Deploy your agents as a web API using BlackSheep:

.. code-block:: python

    from blacksheep import Application, json, get, post
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
    
    app = Application()
    
    # Create an agent
    agent = Agent(
        name="API Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant accessible through an API."
    )
    
    @get("/")
    async def home():
        return json({"message": "Agentle API is running"})
    
    @post("/ask")
    async def ask(request_data: dict):
        query = request_data.get("query")
        if not query:
            return json({"error": "Missing 'query' field"}, status=400)
            
        response = agent.run(query)
        return json({"response": response.text})
    
    # Run with: uvicorn app:app --reload