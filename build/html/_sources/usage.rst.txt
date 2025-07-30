Usage
=====

Basic Usage
----------

Create and use Agentle agents in your Python applications:

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

Core Concepts
-----------

Agents
~~~~~~

The core building block of Agentle is the ``Agent`` class. Each agent:

- Can process various input types (text, images, structured data)
- Can call tools/functions to perform actions
- Can generate structured outputs
- Maintains context through conversations
- Can incorporate static knowledge from documents, URLs, or text

Static Knowledge Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enhance your agents with domain-specific knowledge from various sources:

.. code-block:: python

   from agentle.agents.agent import Agent
   from agentle.agents.knowledge.static_knowledge import StaticKnowledge
   from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

   # Create an agent with static knowledge
   travel_expert = Agent(
       name="Japan Travel Expert",
       generation_provider=GoogleGenerationProvider(),
       model="gemini-2.5-flash",
       instructions="You are a Japan travel expert who provides detailed information about Japanese destinations.",
       # Provide static knowledge from multiple sources
       static_knowledge=[
           # Include knowledge from local documents - cache for 1 hour (3600 seconds)
           StaticKnowledge(content="data/japan_travel_guide.pdf", cache=3600),
           # Include knowledge from websites - cache indefinitely
           StaticKnowledge(content="https://www.japan-guide.com/", cache="infinite"),
           # Include direct text knowledge - no caching (default)
           "Tokyo is the capital of Japan and one of the most populous cities in the world."
       ]
   )

Tools (Function Calling)
~~~~~~~~~~~~~~~~~~~~~~

Extend your agents with custom tools to perform actions beyond text generation:

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
~~~~~~~~~~~~~~~~

Get strongly-typed responses from your agents using Pydantic models:

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

Agent Composition
----------------

Agent Pipelines
~~~~~~~~~~~~~

Connect agents in a sequence where the output of one becomes the input to the next:

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
~~~~~~~~~

Create teams of specialized agents with an orchestrator that dynamically selects the most appropriate agent for each task:

.. code-block:: python

   from agentle.agents.agent import Agent
   from agentle.agents.agent_team import AgentTeam
   from agentle.agents.a2a.models.agent_skill import AgentSkill

   # Create specialized agents with different skills
   research_agent = Agent(
       name="Research Agent",
       description="Specialized in finding accurate information on various topics",
       generation_provider=provider,
       model="gemini-2.5-flash",
       instructions="You are a research agent focused on gathering accurate information.",
       skills=[
           AgentSkill(name="search", description="Find information on any topic"),
           AgentSkill(name="fact-check", description="Verify factual claims"),
       ],
   )

   coding_agent = Agent(
       name="Coding Agent",
       description="Specialized in writing and debugging code",
       generation_provider=provider,
       model="gemini-2.5-flash",
       instructions="You are a coding expert focused on writing clean, efficient code.",
       skills=[
           AgentSkill(name="code-generation", description="Write code in various languages"),
           AgentSkill(name="debugging", description="Find and fix bugs in code"),
       ],
   )

   # Create a team with these agents
   team = AgentTeam(
       agents=[research_agent, coding_agent],
       orchestrator_provider=provider,
       orchestrator_model="gemini-2.5-flash",
   )

   # Run the team with different queries
   research_query = "What are the main challenges in quantum computing today?"
   research_result = team.run(research_query)

Deployment Options
----------------

Web API with BlackSheep
~~~~~~~~~~~~~~~~~~~~

Expose your agent as a RESTful API:

.. code-block:: python

   from agentle.agents.agent import Agent
   from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
   from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

   # Create your agent
   code_assistant = Agent(
       name="Code Assistant",
       description="An AI assistant specialized in helping with programming tasks.",
       generation_provider=GoogleGenerationProvider(),
       model="gemini-2.5-flash",
       instructions="""You are a helpful programming assistant.
       You can answer questions about programming languages, help debug code,
       explain programming concepts, and provide code examples.""",
   )

   # Convert the agent to a BlackSheep ASGI application
   app = AgentToBlackSheepApplicationAdapter().adapt(code_assistant)

   # Run the API server
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="127.0.0.1", port=8000)

Interactive UI with Streamlit
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a chat interface for your agent:

.. code-block:: python

   from agentle.agents.agent import Agent
   from agentle.agents.ui.streamlit import AgentToStreamlit

   # Create your agent
   travel_agent = Agent(
       name="Travel Guide",
       description="A helpful travel guide that answers questions about destinations.",
       generation_provider=GoogleGenerationProvider(),
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

For more detailed examples, see the :doc:`examples` page. 