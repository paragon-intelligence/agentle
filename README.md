# Agentle

<p align="center">
  <img src="/docs/logo.png" alt="Agentle Logo" width="200"/>
</p>

> A powerful yet elegant framework for building the next generation of AI agents.

Agentle makes it effortless to create, compose, and deploy intelligent AI agents - from simple task-focused agents to complex multi-agent systems. Built with developer productivity and type safety in mind, Agentle provides a clean, intuitive API for transforming cutting-edge AI capabilities into production-ready applications.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
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

### Web API with BlackSheep (Experimental)

Expose your agent or A2A interface as a RESTful API:

#### Agent API

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

**Available Endpoints:**
- `POST /api/v1/agents/code_assistant/run` - Send prompts to the agent and get responses synchronously
- `GET /openapi` - Get the OpenAPI specification
- `GET /docs` - Access the interactive API documentation

#### A2A Interface API

For more complex asynchronous workloads, expose your agent using the Agent-to-Agent (A2A) protocol:

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

**Available Endpoints:**
- `POST /api/v1/tasks/send` - Send a task to the agent asynchronously
- `POST /api/v1/tasks/get` - Get task results
- `POST /api/v1/tasks/cancel` - Cancel a running task
- `WebSocket /api/v1/notifications` - Subscribe to push notifications about task status changes (WIP)
- `GET /openapi` - Get the OpenAPI specification
- `GET /docs` - Access the interactive API documentation

The A2A interface provides a message broker pattern for task processing, similar to RabbitMQ, but exposed through a RESTful API interface.

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

### Agent-to-Agent (A2A) Interface

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

#### How Agentle Simplifies A2A Integration

Agentle's A2A implementation handles the complexity of:

- **Task Lifecycle Management**: Automatically manages task creation, execution, and state transitions
- **Thread-Safe Execution**: Uses isolated threads with dedicated event loops to prevent concurrency issues
- **Error Handling**: Provides robust error recovery mechanisms during task execution
- **Standardized Messaging**: Offers a clean interface for creating, sending, and processing A2A messages
- **Session Management**: Maintains conversation history and context across multiple interactions
- **Asynchronous Processing**: Transparently converts asynchronous A2A operations into synchronous methods

The `A2AInterface` class acts as the gateway between your application and any A2A-compliant agent, serving as a unified interface for task management, messaging, and notification handling.

#### Agent Cards

Agentle provides full support for A2A Agent Cards, a standardized JSON format that describes an agent's capabilities, skills, and authentication mechanisms.

Agent Cards are essential for agent discovery and interoperability, making it easy for clients to find the right agent for a specific task. They also provide the information needed to communicate with the agent using the A2A protocol.

**Creating Agent Cards:**

```python
from agentle.agents.agent import Agent
from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create an agent with defined skills
travel_agent = Agent(
    name="Travel Guide",
    description="A helpful travel guide that answers questions about destinations.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a knowledgeable travel guide who helps users plan trips.",
    skills=[
        AgentSkill(
            name="Trip Planning",
            description="Creates personalized travel itineraries",
            tags=["travel", "planning", "itinerary"],
            examples=["Plan a 3-day trip to Tokyo", "What should I see in Paris?"]
        ),
        AgentSkill(
            name="Local Tips",
            description="Provides insider advice for destinations",
            tags=["travel", "local", "tips", "advice"],
            examples=["Best local restaurants in Rome", "Hidden gems in Barcelona"]
        )
    ]
)

# Generate the agent card as a JSON dictionary
agent_card = travel_agent.to_agent_card()

# Save the agent card to a file (as recommended by A2A specification)
import json
with open(".well-known/agent.json", "w") as f:
    json.dump(agent_card, f, indent=2)
```

**Loading Agents from Agent Cards:**

```python
from agentle.agents.agent import Agent

# Load an agent card from a remote source
import requests
response = requests.get("https://example.com/.well-known/agent.json")
agent_card = response.json()

# Create an agent from the card
agent = Agent.from_agent_card(agent_card)

# Use the agent
result = agent.run("What destinations would you recommend for a family vacation?")
print(result.text)
```

Agent Cards are particularly useful for:
- Publishing agent capabilities on the web for discovery
- Creating agent catalogs or marketplaces
- Enabling seamless integration between different agent systems
- Documenting agent interfaces in a standardized format

By following the A2A specification for Agent Cards, Agentle agents can interoperate with any other A2A-compliant system.

#### Advanced A2A Usage

Create collaborative agent ecosystems where specialized agents work together:

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

# Create specialized agents
provider = GoogleGenaiGenerationProvider(api_key=os.environ.get("GOOGLE_API_KEY"))
research_agent = Agent(
    name="Research Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You are a research agent. Find relevant information on topics."
)

writing_agent = Agent(
    name="Writing Agent",
    generation_provider=provider,
    model="gemini-2.0-flash", 
    instructions="You are a writing agent. Craft engaging content from research."
)

# Create A2A interfaces for each agent
research_interface = A2AInterface(agent=research_agent)
writing_interface = A2AInterface(agent=writing_agent)

# Step 1: Send research task to the research agent
research_message = Message(
    role="user", 
    parts=[TextPart(text="Research key innovations in quantum computing")]
)
research_task = research_interface.tasks.send(TaskSendParams(message=research_message))
print(f"Research task created with ID: {research_task.id}")

# Step 2: Wait for research to complete
while True:
    research_result = research_interface.tasks.get(TaskQueryParams(id=research_task.id))
    status = research_result.result.status
    
    if status == TaskState.COMPLETED:
        print("\nResearch completed!")
        research_content = research_result.result.history[1].parts[0].text
        break
    elif status == TaskState.FAILED:
        print(f"Research task failed: {research_result.result.error}")
        exit(1)
    print(f"Research status: {status}")
    time.sleep(1)

# Step 3: Pass research results to writing agent
writing_message = Message(
    role="user",
    parts=[TextPart(text=f"Create an engaging blog post based on this research: {research_content}")]
)
writing_task = writing_interface.tasks.send(TaskSendParams(message=writing_message))
print(f"Writing task created with ID: {writing_task.id}")

# Step 4: Wait for writing to complete
while True:
    writing_result = writing_interface.tasks.get(TaskQueryParams(id=writing_task.id))
    status = writing_result.result.status
    
    if status == TaskState.COMPLETED:
        print("\nBlog post completed!")
        blog_post = writing_result.result.history[1].parts[0].text
        print(f"\nFinal blog post:\n{blog_post}")
        break
    elif status == TaskState.FAILED:
        print(f"Writing task failed: {writing_result.result.error}")
        exit(1)
    print(f"Writing status: {status}")
    time.sleep(1)
```

## 🗓️ Roadmap

- 🗣️ Speech-to-Text and Text-to-Speech modules for voice-enabled agents
- 📚 RAG module for integration with knowledge sources and vector databases
- 🔗 More provider integrations
- 📱 Mobile SDK support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
