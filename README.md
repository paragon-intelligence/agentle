<div align="center">
  <img src="/docs/logo.png" alt="Agentle Logo" width="200"/>
  
  <h3>✨ <em>Elegantly Simple AI Agents for Production</em> ✨</h3>
  
  <p>
    <strong>Build powerful AI agents with minimal code, maximum control</strong>
  </p>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
    <a href="https://badge.fury.io/py/agentle"><img src="https://badge.fury.io/py/agentle.svg" alt="PyPI version"></a>
  </p>

  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-why-agentle">Why Agentle</a> •
    <a href="#-features">Features</a> •
    <a href="#-showcase">Showcase</a> •
    <a href="#-documentation">Docs</a>
  </p>
</div>

---

## 🎯 Why Agentle?

<table>
<tr>
<td width="50%">

### 🚀 **Simple Yet Powerful**
```python
# Just 5 lines to create an AI agent
agent = Agent(
    name="Assistant",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant."
)

response = agent.run("How can I help you?")
```

</td>
<td width="50%">

### 🏗️ **Production Ready**
- 🔍 **Built-in Observability** with Langfuse
- 🌐 **Instant APIs** with automatic documentation
- 💪 **Type-Safe** with full type hints
- 🎯 **Structured Outputs** with Pydantic
- 🔧 **Tool Calling** support out of the box

</td>
</tr>
</table>

## ⚡ Quick Start

### Installation

```bash
pip install agentle
```

### Your First Agent in 30 Seconds

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create a simple agent
agent = Agent(
    name="Quick Start Agent",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant who provides concise, accurate information."
)

# Run the agent
response = agent.run("What are the three laws of robotics?")
print(response.text)
```

## 🌟 Features

<div align="center">

| 🎨 **Beautiful UIs** | 🌐 **Instant APIs** | 📊 **Observability** |
|:---:|:---:|:---:|
| Create chat interfaces with Streamlit in minutes | Deploy RESTful APIs with automatic Scalar docs | Track everything with built-in Langfuse integration |
| ![Streamlit UI](https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92) | ![API Docs](https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6) | ![Tracing](https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7) |

</div>

### 🔥 Core Capabilities

<details>
<summary><b>🤖 Intelligent Agents</b> - Build specialized agents with knowledge, tools, and structured outputs</summary>

```python
from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
from pydantic import BaseModel
from typing import List

# Define structured output
class WeatherForecast(BaseModel):
    location: str
    current_temperature: float
    conditions: str
    forecast: List[str]

# Create a weather tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "New York": "Sunny, 75°F",
        "London": "Rainy, 60°F",
        "Tokyo": "Cloudy, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")

# Build the agent
weather_agent = Agent(
    name="Weather Assistant",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a weather forecasting assistant.",
    # Add domain knowledge
    static_knowledge=[
        StaticKnowledge(content="weather_data/climate_patterns.pdf", cache=3600),
        "A heat wave is defined as a period of abnormally hot weather generally lasting more than two days."
    ],
    # Add tools
    tools=[get_weather],
    # Ensure structured responses
    response_schema=WeatherForecast
)

# Get typed responses
response = weather_agent.run("What's the weather like in Tokyo?")
forecast = response.parsed
print(f"Weather in {forecast.location}: {forecast.current_temperature}°C, {forecast.conditions}")
```
</details>

<details>
<summary><b>🔗 Agent Pipelines</b> - Chain agents for complex workflows</summary>

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create provider for reuse
provider = GoogleGenerationProvider()

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
</details>

<details>
<summary><b>👥 Agent Teams</b> - Dynamic orchestration with intelligent routing</summary>

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_team import AgentTeam
from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create provider for reuse
provider = GoogleGenerationProvider()

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
</details>

<details>
<summary><b>🔌 MCP Integration</b> - Connect to external tools via Model Context Protocol</summary>

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer
from agentle.mcp.session_management import RedisSessionManager

# Set up provider
provider = GoogleGenerationProvider()

# Create MCP servers
stdio_server = StdioMCPServer(
    server_name="File System MCP",
    command="/path/to/filesystem_mcp_server",  # Replace with actual command
    server_env={"DEBUG": "1"},
)

# For development (single-process environments)
sse_server_dev = StreamableHTTPMCPServer(
    server_name="Weather API MCP",
    server_url="http://localhost:3000",  # Replace with actual server URL
)

# For production (multi-process environments)
redis_session = RedisSessionManager(
    redis_url="redis://redis-server:6379/0",
    key_prefix="agentle_mcp:",
    expiration_seconds=3600  # 1 hour session lifetime
)

sse_server_prod = StreamableHTTPMCPServer(
    server_name="Weather API MCP",
    server_url="https://api.example.com",
    session_manager=redis_session
)

# Create agent with MCP servers
agent = Agent(
    name="MCP-Augmented Assistant",
    description="An assistant that can access external tools via MCP",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant with access to external tools.",
    mcp_servers=[stdio_server, sse_server_dev],
)

# Use the start_mcp_servers context manager for proper connection handling
with agent.start_mcp_servers():
    # Query that uses MCP server tools
    response = agent.run("What's the weather like in Tokyo today?")
    print(response.generation.text)
```
</details>

## 🖼️ Visual Showcase

### 🎨 Build Beautiful Chat UIs

Transform your agent into a professional chat interface with just a few lines:

```python
from agentle.agents.agent import Agent
from agentle.agents.ui.streamlit import AgentToStreamlit
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create your agent
travel_agent = Agent(
    name="Travel Guide",
    description="A helpful travel guide that answers questions about destinations.",
    generation_provider=GoogleGenerationProvider(),
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

<img width="100%" alt="Streamlit Chat Interface" src="https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92" />

### 🌐 Deploy Production APIs

Expose your agents as RESTful APIs with automatic documentation:

```python
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create your agent
code_assistant = Agent(
    name="Code Assistant",
    description="An AI assistant specialized in helping with programming tasks.",
    generation_provider=GoogleGenerationProvider(),
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

<img width="100%" alt="API Documentation" src="https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6" />

### 📊 Enterprise-Grade Observability

Monitor every aspect of your agents in production:

```python
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create a tracing client
tracing_client = LangfuseObservabilityClient()

# Create an agent with tracing enabled
agent = Agent(
    name="Traceable Agent",
    generation_provider=GoogleGenerationProvider(tracing_client=tracing_client),
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

<img width="100%" alt="Observability Dashboard" src="https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7" />

<img width="100%" alt="Detailed Trace View" src="https://github.com/user-attachments/assets/c38429db-982c-4158-864f-f03e7118618e" />

**Automatic Scoring System** tracks:
- 🎯 **Model Tier Score** - Evaluates the capability tier of the model used
- 🔧 **Tool Usage Score** - Measures how effectively the agent uses available tools  
- 💰 **Token Efficiency Score** - Analyzes the balance between input and output tokens
- ⚡ **Cost Efficiency Score** - Tracks the cost-effectiveness of each generation

<img width="100%" alt="Trace Scores" src="https://github.com/user-attachments/assets/f0aab337-ead3-417b-97ef-0126c833d347" />

## 🏗️ Real-World Examples

### 💬 Customer Support Agent

```python
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Define support tools
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base."""
    # Implementation would search your KB
    return "Found solution: Reset password via email link"

def create_ticket(issue: str, priority: str = "medium") -> str:
    """Create a support ticket."""
    # Implementation would create ticket in your system
    return f"Ticket created with ID: SUPP-12345"

# Create support agent
support_agent = Agent(
    name="Support Hero",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are an empathetic customer support specialist.",
    tools=[search_knowledge_base, create_ticket],
    static_knowledge=["support_policies.pdf", "faq.md"]
)

# Deploy as API
api = AgentToBlackSheepApplicationAdapter().adapt(support_agent)
```

### 📊 Data Analysis Pipeline

```python
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

provider = GoogleGenerationProvider()

# Create specialized agents
data_cleaner = Agent(
    name="Data Cleaner",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You clean and preprocess data, handling missing values and outliers."
)

statistician = Agent(
    name="Statistician",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You perform statistical analysis and identify significant patterns."
)

visualizer = Agent(
    name="Visualizer",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You create clear descriptions of data visualizations and insights."
)

# Build analysis pipeline
analysis_pipeline = AgentPipeline(
    agents=[data_cleaner, statistician, visualizer]
)

# Process data
result = analysis_pipeline.run("Analyze this sales data: Q1: $1.2M, Q2: $1.5M, Q3: $1.1M, Q4: $2.1M")
```

### 🌍 Multi-Provider Resilience

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.failover.failover_generation_provider import FailoverGenerationProvider
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
from agentle.generations.providers.openai.openai import OpenaiGenerationProvider
from agentle.generations.providers.cerebras.cerebras_generation_provider import CerebrasGenerationProvider

# Never go down - automatically failover between providers
resilient_provider = FailoverGenerationProvider(
    generation_providers=[
        GoogleGenerationProvider(),
        OpenaiGenerationProvider(api_key="your-openai-key"),
        CerebrasGenerationProvider(api_key="your-cerebras-key")
    ],
    shuffle=True
)

agent = Agent(
    name="Resilient Assistant",
    generation_provider=resilient_provider,
    # Use ModelKind instead of specific model names for better compatibility
    model="category_pro",  # Each provider maps this to their equivalent model
    instructions="You are a helpful assistant."
)
```

#### 🧠 Using ModelKind for Provider Abstraction

Agentle provides a powerful abstraction layer with `ModelKind` that decouples your code from specific provider model names:

```python
# Instead of hardcoding provider-specific model names:
agent = Agent(generation_provider=provider, model="gpt-4o")  # Only works with OpenAI

# Use ModelKind for provider-agnostic code:
agent = Agent(generation_provider=provider, model="category_pro")  # Works with any provider
```

**Benefits of ModelKind:**

- **Provider independence**: Write code that works with any AI provider
- **Future-proof**: When providers release new models, only mapping tables need updates
- **Capability-based selection**: Choose models based on capabilities, not names
- **Perfect for failover**: Each provider automatically maps to its equivalent model
- **Consistency**: Standardized categories across all providers

Each provider implements `map_model_kind_to_provider_model()` to translate these abstract categories to their specific models (e.g., "category_pro" → "gpt-4o" for OpenAI or "gemini-2.5-pro" for Google).

## 🛠️ Advanced Features

### 🎭 Flexible Input Types

Agentle agents handle any input type seamlessly:

```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
import pandas as pd
from PIL import Image
import numpy as np

# Create a basic agent
agent = Agent(
    generation_provider=GoogleGenerationProvider(),
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

### 🧩 Prompt Management

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

### 💬 Rich Messaging System

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

## 📚 Full Feature Documentation

### 🔗 Agent-to-Agent (A2A) Protocol

Agentle provides built-in support for Google's [A2A Protocol](https://google.github.io/A2A/):

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
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Set up agent and A2A interface
provider = GoogleGenerationProvider(api_key=os.environ.get("GOOGLE_API_KEY"))
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

### 🔧 Tool Calling and Structured Outputs Combined

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
    generation_provider=GoogleGenerationProvider(),
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

### 📄 Custom Document Parsers

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
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You analyze documents with precision.",
    static_knowledge=[
        StaticKnowledge(content="contracts/agreement.pdf", cache="infinite")
    ],
    # Pass your custom parser to the agent
    document_parser=CustomParser()
)
```

## 🧠 Philosophy

> **"Simplicity is the ultimate sophistication"** - Leonardo da Vinci

I created Agentle out of frustration with the direction of other agent frameworks. Many frameworks have lost sight of clean design principles by adding numerous configuration flags to their Agent constructors (like ``enable_whatever=True``, ``add_memory=True``, etc.). This approach creates countless possible combinations, making debugging and development unnecessarily complex. 

Also, there is a lot of market pressure that **unfortunately** leads the devs to push unpolished stuff into prod, because their framework must always be on the top of the frameworks. That's not the case right here. I made this for myself, but it might be helpful to other devs as well. I am a solo developer in this framework (for now), but I want to only ship stuff that developers will really need. And to ship stuff only when it's ready (e.g PROPERLY TYPED, since many frameworks just goes to **kwargs or "Any" in many cases).

I wanted to create a framework that was both helpful in some common scenarios, but let the developer do his job as well.

Agentle strives to maintain a careful balance between simplicity and practicality. For example, I've wrestled with questions like whether document parsing functionality belongs in the Agent constructor. While not "simple" in the purest sense, such features can be practical for users. Finding this balance is central to Agentle's design philosophy.

Core principles of Agentle:

* Avoiding configuration flags in constructors whenever possible
* Organizing each class and function in separate modules by design
* Following the Single Responsibility Principle rather than strictly Pythonic conventions (5000 SLOC types.py file)
* Creating a codebase that's not only easy to use but also easy to maintain and extend (though the limitations of python about circular imports, me (and other devs), should be aware of this issue when working with one class per module)

Through this thoughtful approach to architecture, Agentle aims to provide a framework that's both powerful and elegant for building the next generation of AI agents.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>
    <strong>Built with ❤️ by a developer, for developers</strong>
  </p>
  <p>
    <a href="#-agentle">⬆ Back to top</a>
  </p>
</div>
