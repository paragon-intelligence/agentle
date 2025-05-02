# Agentle

<p align="center">
  <img src="/docs/logo.png" alt="Agentle Logo" width="200"/>
</p>

> A powerful yet elegant framework for building the next generation of AI agents.

Agentle makes it effortless to create, compose, and deploy intelligent AI agents - from simple task-focused agents to complex multi-agent systems. Built with developer productivity and type safety in mind, Agentle provides a clean, intuitive API for transforming cutting-edge AI capabilities into production-ready applications.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/agentle.svg)](https://badge.fury.io/py/agentle)

## ✨ Key Features

- 🧠 **Simple Agent Creation** - Build powerful AI agents with minimal code
- 🔄 **Composable Architecture** - Create sequential pipelines or dynamic teams of specialized agents
- 🛠️ **Tool Integration** - Seamlessly connect agents to external tools and functions
- 📊 **Structured Outputs** - Get strongly-typed responses with Pydantic integration
- 🌐 **Ready for Production** - Deploy as APIs (BlackSheep), UIs (Streamlit), or embedded in apps
- 🔍 **Built-in Observability** - Automatic tracing via Langfuse with extensible interfaces
- 🤝 **Agent-to-Agent (A2A)** - Support for Google's standardized A2A protocol
- 📝 **Prompt Management** - Flexible system for organizing and managing prompts

## 📦 Installation

```bash
pip install agentle
```

## 🚀 Quick Start

Create a simple agent in just a few lines of code:

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

## 🧩 Core Concepts

### Agents

The core building block of Agentle is the `Agent` class. Each agent:

- Can process various input types (text, images, structured data)
- Can call tools/functions to perform actions
- Can generate structured outputs
- Maintains context through conversations

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create an agent with specific instructions
travel_agent = Agent(
    name="Travel Guide",
    description="A helpful travel guide that answers questions about destinations.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a knowledgeable travel guide who helps users plan trips.
    You provide information about destinations, offer travel tips, suggest itineraries,
    and answer questions about local customs, attractions, and practical travel matters."""
)

# Run the agent
result = travel_agent.run("What are the must-see attractions in Tokyo?")
print(result.text)
```

### Tools (Function Calling)

Extend your agents with custom tools to perform actions beyond text generation:

```python
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
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant that can answer questions about the weather.",
    tools=[get_weather]  # Pass the function as a tool
)

# The agent will automatically use the tool when appropriate
response = weather_agent.run("What's the weather like in Tokyo?")
print(response.text)
```

### Structured Outputs

Get strongly-typed responses from your agents using Pydantic models:

```python
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
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
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
```

## 🌈 Flexible Input Types

Agentle agents can process an incredible variety of input types out-of-the-box, making it simple to work with different data formats without complex conversions:

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from io import StringIO, BytesIO

# Create a basic agent
agent = Agent(
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a versatile assistant that can analyze different types of data."
)

# String input (simplest case)
agent.run("What is the capital of Japan?")

# Pandas DataFrame
df = pd.DataFrame({
    "Country": ["Japan", "France", "USA"],
    "Capital": ["Tokyo", "Paris", "Washington DC"],
    "Population": [126.3, 67.8, 331.9]
})
agent.run(df)  # Automatically converts to markdown table

# Image input (for multimodal models)
img = Image.open("chart.png")
agent.run(img)  # Automatically handles image format

# NumPy array
data = np.array([[1, 2, 3], [4, 5, 6]])
agent.run(data)  # Automatically formats array

# Dictionary/JSON
user_data = {
    "name": "Alice",
    "interests": ["AI", "Python", "Data Science"],
    "experience_years": 5
}
agent.run(user_data)  # Automatically formats as JSON

# Date and time
agent.run(datetime.now())  # Formatted as ISO string

# File path
agent.run(Path("report.txt"))  # Reads and processes file content

# Pydantic model
class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str]

profile = UserProfile(name="Bob", age=28, interests=["AI", "Robotics"])
agent.run(profile)  # Automatically formats model as JSON

# File-like objects
text_io = StringIO("This is some text data from a stream")
agent.run(text_io)  # Reads content from StringIO
```

## 📨 Message and Part Types

Agentle provides a rich messaging system that enables fine-grained control over how you communicate with agents. This is especially powerful for multimodal interactions and complex conversations:

### Message Types

```python
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.message_parts.text import TextPart

# Create a conversation with multiple message types
messages = [
    # System instructions (not visible to the user)
    DeveloperMessage(parts=[
        TextPart(text="You are a helpful travel assistant that speaks in a friendly tone.")
    ]),
    
    # User's initial message
    UserMessage(parts=[
        TextPart(text="I'm planning a trip to Japan in April.")
    ]),
    
    # Previous assistant response in the conversation
    AssistantMessage(parts=[
        TextPart(text="That's a wonderful time to visit Japan! Cherry blossoms should be in bloom.")
    ]),
    
    # User's follow-up question
    UserMessage(parts=[
        TextPart(text="What cities should I visit for the best cherry blossom viewing?")
    ])
]

# Pass the complete conversation to the agent
result = agent.run(messages)
```

### Part Types

Each message can contain multiple parts of different types, enabling rich multimodal interactions:

```python
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.tools.tool import Tool

# Create a message with different part types
message = UserMessage(
    parts=[
        # Text part for regular text input
        TextPart(text="Can you analyze this image and data?"),
        
        # File part for image analysis (multimodal models)
        FilePart(
            data=open("vacation_photo.jpg", "rb").read(),
            mime_type="image/jpeg"
        ),
        
        # Tool reference to pass tool execution results
        Tool.from_callable(get_weather).with_result(
            args={"location": "Tokyo"},
            result="Sunny, 23°C, Humidity: 45%"
        )
    ]
)

# Run the agent with the multi-part message
result = agent.run(message)
```

### Context Object

For maximum control, you can create a Context object to manage complete conversations:

```python
from agentle.agents.context import Context
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.agents.step import Step

# Create a custom context with specific messages
context = Context(
    messages=[
        DeveloperMessage(parts=[
            TextPart(text="You are a travel planning assistant with expertise in budgeting.")
        ]),
        UserMessage(parts=[
            TextPart(text="I want to plan a 7-day trip to Europe with a $3000 budget.")
        ])
    ],
    # Optionally track conversation steps
    steps=[
        Step(type="user_input", content="Initial travel budget query")
    ]
)

# Run the agent with the custom context
result = agent.run(context)
```

This messaging system allows for precise control over agent interactions, enabling everything from simple queries to complex multi-turn, multimodal conversations with full context management.

## 🔄 Agent Composition

### Agent Pipelines

Connect agents in a sequence where the output of one becomes the input to the next:

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline

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

### Agent Teams

Create teams of specialized agents with an orchestrator that dynamically selects the most appropriate agent for each task:

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_team import AgentTeam
from agentle.agents.a2a.models.agent_skill import AgentSkill

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

## 🌐 Deployment Options

### Web API with BlackSheep

Expose your agent as a RESTful API:

```python
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application import AgentToBlackSheepApplication

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
app = AgentToBlackSheepApplication().adapt(code_assistant)

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Interactive UI with Streamlit

Create a chat interface for your agent:

```python
from agentle.agents.agent import Agent
from agentle.agents.ui.streamlit import AgentToStreamlit

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

## 🔍 Observability and Tracing

Agentle provides built-in observability through Langfuse, with a flexible interface for other providers:

```python
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
from agentle.agents.agent import Agent

# Create a tracing client
tracer = LangfuseObservabilityClient()

# Create an agent with tracing enabled
agent = Agent(
    name="Traceable Agent",
    generation_provider=GoogleGenaiGenerationProvider(),
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

## 📝 Prompt Management

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

## 🧪 Tool Calling and Structured Outputs Combined

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

## 🔄 Agent-to-Agent (A2A) Interface

Agentle implements Google's A2A Protocol, allowing agents to communicate with each other using standardized interfaces regardless of their underlying implementations or hosting environments.

```python
from agentle.agents.agent import Agent
from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create two agents
agent1 = Agent(
    name="Task Creator",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You create tasks for other agents to complete."
)

agent2 = Agent(
    name="Task Executor",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You execute tasks that other agents assign to you."
)

# Initialize A2A interface for agent2
a2a = A2AInterface(agent=agent2)

# Create a message to send to agent2
message = Message(
    role="user",
    parts=[TextPart(text="Please summarize the key features of Python 3.12")]
)

# Send a task to agent2
task_params = TaskSendParams(
    message=message,
    sessionId="session-123"
)

# Send the task
task = a2a.tasks.send(task_params)
print(f"Task created with ID: {task.id}")
print(f"Initial status: {task.status}")

# Get the result when ready
result = a2a.tasks.get({"id": task.id})
print(f"Final status: {result.task.status}")
print(f"Response: {result.task.response.message.parts[0].text}")
```

The A2A protocol provides several key benefits:
- **Interoperability**: Agents can communicate regardless of their underlying implementation
- **Standardization**: Common interface for task creation, monitoring, and cancellation
- **Session Management**: Track conversations across multiple interactions
- **Asynchronous Processing**: Submit tasks and retrieve results when ready
- **Extensibility**: Support for various message types including text, images, and custom data formats

## 🗓️ Roadmap

- 🗣️ Speech-to-Text and Text-to-Speech modules for voice-enabled agents
- 📚 RAG module for integration with knowledge sources and vector databases
- 🔗 More provider integrations
- 📱 Mobile SDK support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
