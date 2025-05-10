# Agentle

<p align="center">
  <img src="/docs/logo.png" alt="Agentle Logo" width="200"/>
</p>

<p align="center">
  <b>A powerful yet elegant framework for building the next generation of AI agents</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://badge.fury.io/py/agentle"><img src="https://badge.fury.io/py/agentle.svg" alt="PyPI version"></a>
</p>

<p align="center">
  <b>🚀 OFFICIAL RELEASE: MAY 16 - 2025 🚀</b>
</p>

---

Agentle makes it effortless to create, compose, and deploy intelligent AI agents - from simple task-focused agents to complex multi-agent systems. Built with developer productivity and type safety in mind, Agentle provides a clean, intuitive API for transforming cutting-edge AI capabilities into production-ready applications.

## ⚡ Quick Start

### Installation

```bash
pip install agentle
```

### Create Your First Agent in Seconds

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create a simple agent
agent = Agent(
    name="Quick Start Agent",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant who provides concise, accurate information."
)

# Run the agent
response = agent.run("What are the three laws of robotics?")

# Print the response
print(response.text)
```

## 🌈 Visual Showcase

### Interactive Chat UI with Streamlit

Create beautiful chat interfaces with just a few lines of code:

```python
from agentle.agents.agent import Agent
from agentle.agents.ui.streamlit import AgentToStreamlit
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create your agent
travel_agent = Agent(
    name="Travel Guide",
    description="A helpful travel guide that answers questions about destinations.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
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
```

<p align="center">
  <img width="1000" alt="Streamlit Interface" src="https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92" />
</p>

### Production-Ready API with BlackSheep

Expose your agent as a RESTful API with automatic Scalar documentation:

```python
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create your agent
code_assistant = Agent(
    name="Code Assistant",
    description="An AI assistant specialized in helping with programming tasks.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
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
```

<p align="center">
  <img width="1000" alt="API Documentation" src="https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6" />
</p>

### Enterprise-Grade Observability

Monitor your agents in production with built-in tracing and observability:

```python
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create a tracing client
tracing_client = LangfuseObservabilityClient()

# Create an agent with tracing enabled
agent = Agent(
    name="Traceable Agent",
    generation_provider=GoogleGenaiGenerationProvider(tracing_client=tracing_client),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant.",
    # Tracing is automatically enabled
)

# Run the agent - tracing happens automatically
response = agent.run(
    "What's the weather in Tokyo?", 
    trace_params={
        "name": "weather_query",
        "user_id": "user123",
        "metadata": {"source": "mobile_app"}
    }
)
```

<p align="center">
  <img width="1000" alt="Tracing Dashboard" src="https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7" />
</p>

<p align="center">
  <img width="1000" alt="Detailed Trace View" src="https://github.com/user-attachments/assets/c38429db-982c-4158-864f-f03e7118618e" />
</p>

## ✨ Key Features

- 🧠 **Simple Agent Creation** - Build powerful AI agents with minimal code
- 🔄 **Composable Architecture** - Create sequential pipelines or dynamic teams of specialized agents
- 🛠️ **Tool Integration** - Seamlessly connect agents to external tools and functions
- 📊 **Structured Outputs** - Get strongly-typed responses with Pydantic integration
- 🌐 **Ready for Production** - Deploy as APIs (BlackSheep), UIs (Streamlit), or embedded in apps
- 🔍 **Built-in Observability** - Automatic tracing via Langfuse with extensible interfaces
- 🤝 **Agent-to-Agent (A2A)** - Support for Google's standardized A2A protocol
- 📝 **Prompt Management** - Flexible system for organizing and managing prompts
- 📚 **Knowledge Integration** - Seamlessly incorporate static knowledge from various sources

## 🧩 Core Concepts

### Intelligent Agents

Build specialized agents with domain knowledge, tools, and structured outputs:

```python
from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider
from pydantic import BaseModel
from typing import List

# Define structured output schema
class WeatherForecast(BaseModel):
    location: str
    current_temperature: float
    conditions: str
    forecast: List[str]

# Define a weather lookup tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "New York": "Sunny, 75°F",
        "London": "Rainy, 60°F",
        "Tokyo": "Cloudy, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")

# Create a powerful agent that combines knowledge, tools, and structured output
weather_agent = Agent(
    name="Weather Assistant",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a weather forecasting assistant.",
    # Add domain-specific knowledge
    static_knowledge=[
        StaticKnowledge(content="weather_data/climate_patterns.pdf", cache=3600),
        "A heat wave is defined as a period of abnormally hot weather generally lasting more than two days."
    ],
    # Add function calling capabilities
    tools=[get_weather],
    # Ensure structured responses
    response_schema=WeatherForecast
)

# Run the agent
response = weather_agent.run("What's the weather like in Tokyo?")

# Access structured data with type hints
forecast = response.parsed
print(f"Weather in {forecast.location}: {forecast.current_temperature}°C, {forecast.conditions}")
```

### Agent Composition

#### Agent Pipelines

Connect agents in a sequence where output of one becomes input to the next:

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create provider for reuse
provider = GoogleGenaiGenerationProvider()

# Create specialized agents
research_agent = Agent(
    name="Research Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a research agent focused on gathering information.
    Be thorough and prioritize accuracy over speculation."""
)

analysis_agent = Agent(
    name="Analysis Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are an analysis agent that identifies patterns.
    Highlight meaningful relationships and insights from the data."""
)

summary_agent = Agent(
    name="Summary Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
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
```

#### Agent Teams

Create teams of specialized agents with an orchestrator that dynamically selects the most appropriate agent for each task:

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_team import AgentTeam
from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create provider for reuse
provider = GoogleGenaiGenerationProvider()

# Create specialized agents with different skills
research_agent = Agent(
    name="Research Agent",
    description="Specialized in finding accurate information on various topics",
    generation_provider=provider,
    model="gemini-2.0-flash",
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
    model="gemini-2.0-flash",
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
    orchestrator_model="gemini-2.0-flash",
)

# Run the team with different queries
research_query = "What are the main challenges in quantum computing today?"
research_result = team.run(research_query)

coding_query = "Write a Python function to find the Fibonacci sequence up to n terms."
coding_result = team.run(coding_query)
```

## 🌐 Advanced Deployment Options

### A2A Protocol Integration

Leverage Google's Agent-to-Agent protocol for enterprise integration:

```python
from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create your agent
code_assistant = Agent(
    name="Async Code Assistant",
    description="An AI assistant specialized in helping with programming tasks asynchronously.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a helpful programming assistant.
    You can answer questions about programming languages, help debug code,
    explain programming concepts, and provide code examples.""",
)

# Create an A2A interface for the agent
a2a_interface = A2AInterface(agent=code_assistant)

# Convert the A2A interface to a BlackSheep ASGI application
app = AgentToBlackSheepApplicationAdapter().adapt(a2a_interface)

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

<p align="center">
  <img width="1000" alt="A2A Interface Documentation" src="https://github.com/user-attachments/assets/b818954f-fb58-4079-98b2-88bc9d84b2c4" />
</p>

### Advanced Knowledge Integration

Enhance your agents with domain-specific knowledge from various sources:

```python
from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create an agent with static knowledge
travel_expert = Agent(
    name="Japan Travel Expert",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
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

# The agent will incorporate the knowledge when answering
response = travel_expert.run("What should I know about visiting Tokyo in cherry blossom season?")
print(response.text)
```

## 🧪 Advanced Features

### Flexible Input Types

Agentle agents can process various input types without complex conversions:

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider
import pandas as pd
from PIL import Image
import numpy as np

# Create a basic agent
agent = Agent(
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a versatile assistant that can analyze different types of data."
)

# Process different input types
agent.run("What is the capital of Japan?")  # String

# DataFrame input
df = pd.DataFrame({
    "Country": ["Japan", "France", "USA"],
    "Capital": ["Tokyo", "Paris", "Washington DC"],
    "Population": [126.3, 67.8, 331.9]
})
agent.run(df)  # Automatically converts to markdown table

# Image input (for multimodal models)
img = Image.open("chart.png")
agent.run(img)  # Automatically handles image format

# Dictionary/JSON
user_data = {"name": "Alice", "interests": ["AI", "Python"]}
agent.run(user_data)  # Automatically formats as JSON
```

### Prompt Management

Manage prompts with a flexible prompt provider system:

```python
from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.fs_prompt_provider import FSPromptProvider

# Create a prompt provider that loads prompts from files
prompt_provider = FSPromptProvider(base_path="./prompts")

# Load a prompt
weather_prompt = prompt_provider.provide("weather_template")

# Compile the prompt with variables
compiled_prompt = weather_prompt.compile(
    location="Tokyo",
    units="celsius",
    days=5
)

# Use the prompt with an agent
agent.run(compiled_prompt)
```

### Rich Messaging System

Create multimodal conversations with fine-grained control:

```python
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.file import FilePart

# Create a conversation with multiple message types
messages = [
    # System instructions (not visible to the user)
    DeveloperMessage(parts=[
        TextPart(text="You are a helpful travel assistant that speaks in a friendly tone.")
    ]),
    
    # User's initial message with image
    UserMessage(parts=[
        TextPart(text="What can you tell me about this landmark?"),
        FilePart(
            data=open("landmark_photo.jpg", "rb").read(),
            mime_type="image/jpeg"
        )
    ]),
    
    # Previous assistant response in the conversation
    AssistantMessage(parts=[
        TextPart(text="That's the famous Tokyo Tower in Japan!")
    ]),
    
    # User's follow-up question
    UserMessage(parts=[
        TextPart(text="What's the best time to visit?")
    ])
]

# Pass the complete conversation to the agent
result = agent.run(messages)
```

## 🧠 Author's Note

I created Agentle out of frustration with the direction of other agent frameworks. Many frameworks have lost sight of clean design principles by adding numerous configuration flags to their Agent constructors (like ``enable_whatever=True``, ``add_memory=True``, etc.). This approach creates countless possible combinations, making debugging and development unnecessarily complex. Also, there is a lot of market pressure that **unfortunately** leads the devs to push unpolished stuff into prod, because their framework must always be on the top of the frameworks. That's not the case right here. I made this for myself, but it might be helpful to other devs as well. I am a solo developer in this framework (for now), but I want to only ship stuff that developers will really need. And to ship stuff only when it's ready (e.g PROPERLY TYPED, since many frameworks just goes to **kwargs or "Any" in many cases).

I wanted to create a framework that was both helpful in some common scenarios, but let the developer do his job as well.

Agentle strives to maintain a careful balance between simplicity and practicality. For example, I've wrestled with questions like whether document parsing functionality belongs in the Agent constructor. While not "simple" in the purest sense, such features can be practical for users. Finding this balance is central to Agentle's design philosophy.

Core principles of Agentle:

* Avoiding configuration flags in constructors whenever possible
* Organizing each class and function in separate modules by design
* Following the Single Responsibility Principle rather than strictly Pythonic conventions (5000 SLOC types.py file)
* Creating a codebase that's not only easy to use but also easy to maintain and extend (though the limitations of python about circular imports, me (and other devs), should be aware of this issue when working with one class per module)

Through this thoughtful approach to architecture, Agentle aims to provide a framework that's both powerful and elegant for building the next generation of AI agents.

---

## 📚 Full Feature Documentation

### Agent-to-Agent (A2A) Protocol

Agentle provides built-in support for Google's [A2A Protocol](https://google.github.io/A2A/), enabling seamless communication between agents:

```python
import os
import time

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Set up agent and A2A interface
provider = GoogleGenaiGenerationProvider(api_key=os.environ.get("GOOGLE_API_KEY"))
agent = Agent(name="Example Agent", generation_provider=provider, model="gemini-2.0-flash")
a2a = A2AInterface(agent=agent)

# Send task to agent
message = Message(role="user", parts=[TextPart(text="What are three facts about the Moon?")])
task = a2a.tasks.send(TaskSendParams(message=message))
print(f"Task sent with ID: {task.id}")

# Wait for task completion and get result
while True:
    result = a2a.tasks.get(TaskQueryParams(id=task.id))
    status = result.result.status
    
    if status == TaskState.COMPLETED:
        print("\nResponse:", result.result.history[1].parts[0].text)
        break
    elif status == TaskState.FAILED:
        print(f"Task failed: {result.result.error}")
        break
    print(f"Status: {status}")
    time.sleep(1)
```

#### What is A2A and Why It Matters

**A2A (Agent-to-Agent)** is an open protocol designed to enable standardized communication between autonomous agents built on different frameworks and by various vendors. Key benefits include:

- **Interoperability**: Agents built with different frameworks can communicate seamlessly
- **Enterprise Integration**: Easily integrate agents into existing enterprise applications
- **Asynchronous Communication**: Non-blocking task management for long-running operations
- **State Management**: Track task progress and history across agent interactions
- **Multimodal Support**: Exchange rich content including text, images, and structured data
- **Open Standard**: Community-driven protocol designed for widespread adoption

### Tool Calling and Structured Outputs Combined

For even more powerful agents, combine tool calling with structured outputs:

```python
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
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
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
```

### Custom Document Parsers

For specialized knowledge extraction needs, you can provide a custom document parser to the agent:

```python
from typing import override
from pathlib import Path
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent

# Create a custom parser
class CustomParser(DocumentParser):
    """Parser with specialized document understanding"""
    
    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        # Read the document file
        path = Path(document_path)
        file_content = path.read_text(encoding="utf-8")
        
        # Use your custom parsing logic
        parsed_content = file_content.upper()  # Simple example transformation
        
        # Return in the standard ParsedDocument format
        return ParsedDocument(
            name=path.name,
            sections=[
                SectionContent(
                    number=1,
                    text=parsed_content,
                    md=parsed_content
                )
            ]
        )

# Use the custom parser with an agent
from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge

agent = Agent(
    name="Document Expert",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You analyze documents with precision.",
    static_knowledge=[
        StaticKnowledge(content="contracts/agreement.pdf", cache="infinite")
    ],
    # Pass your custom parser to the agent
    document_parser=CustomParser()
)
