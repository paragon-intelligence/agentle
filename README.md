# 🤖 Agentle

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
    <a href="#"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  </p>

  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-why-agentle">Why Agentle</a> •
    <a href="#-features">Features</a> •
    <a href="#-showcase">Showcase</a> •
    <a href="#-documentation">Docs</a> •
    <a href="#-community">Community</a>
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
    generation_provider=GoogleGenaiGenerationProvider(),
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
from agentle import Agent
from agentle.providers import GoogleGenaiGenerationProvider

# Create a simple agent
agent = Agent(
    name="Quick Start Agent",
    generation_provider=GoogleGenaiGenerationProvider(),
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
from agentle import Agent, StaticKnowledge
from pydantic import BaseModel

# Define structured output
class WeatherForecast(BaseModel):
    location: str
    temperature: float
    conditions: str

# Create a weather tool
def get_weather(location: str) -> str:
    return f"Sunny, 75°F in {location}"

# Build the agent
weather_agent = Agent(
    name="Weather Assistant",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a weather forecasting assistant.",
    # Add domain knowledge
    static_knowledge=[
        StaticKnowledge(content="weather_data.pdf", cache=3600),
        "Heat waves last more than two days."
    ],
    # Add tools
    tools=[get_weather],
    # Ensure structured responses
    response_schema=WeatherForecast
)

# Get typed responses
result = weather_agent.run("What's the weather in Tokyo?")
print(f"Weather in {result.parsed.location}: {result.parsed.temperature}°C")
```
</details>

<details>
<summary><b>🔗 Agent Pipelines</b> - Chain agents for complex workflows</summary>

```python
from agentle import AgentPipeline

# Create specialized agents
research_agent = Agent(name="Researcher", ...)
analysis_agent = Agent(name="Analyst", ...)
summary_agent = Agent(name="Summarizer", ...)

# Chain them together
pipeline = AgentPipeline(
    agents=[research_agent, analysis_agent, summary_agent],
    debug_mode=True
)

# Run the pipeline
result = pipeline.run("Research AI impact on healthcare")
```
</details>

<details>
<summary><b>👥 Agent Teams</b> - Dynamic orchestration with intelligent routing</summary>

```python
from agentle import AgentTeam

# Create a team of specialists
team = AgentTeam(
    agents=[research_agent, coding_agent, writing_agent],
    orchestrator_provider=provider,
    orchestrator_model="gemini-2.0-flash"
)

# The orchestrator automatically routes to the right agent
result = team.run("Write a Python function to analyze stock data")
```
</details>

<details>
<summary><b>🔌 MCP Integration</b> - Connect to external tools via Model Context Protocol</summary>

```python
from agentle.mcp import StdioMCPServer, StreamableHTTPMCPServer

# Connect to local MCP servers
stdio_server = StdioMCPServer(
    server_name="File System",
    command="/path/to/filesystem_mcp"
)

# Connect to remote MCP servers
http_server = StreamableHTTPMCPServer(
    server_name="Weather API",
    server_url="https://api.example.com"
)

# Create agent with MCP capabilities
agent = Agent(
    name="MCP-Powered Assistant",
    mcp_servers=[stdio_server, http_server],
    ...
)

# Tools are automatically available
with agent.start_mcp_servers():
    response = agent.run("What files are in the current directory?")
```
</details>

## 🖼️ Visual Showcase

### 🎨 Build Beautiful Chat UIs

Transform your agent into a professional chat interface with just a few lines:

```python
from agentle.ui import AgentToStreamlit

# Convert any agent to a Streamlit app
app = AgentToStreamlit(
    title="AI Assistant",
    description="Your personal AI helper",
    initial_mode="presentation"
).adapt(your_agent)

# Run it!
app()
```

<img width="100%" alt="Streamlit Chat Interface" src="https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92" />

### 🌐 Deploy Production APIs

Expose your agents as RESTful APIs with automatic documentation:

```python
from agentle.asgi import AgentToBlackSheepApplicationAdapter

# Convert to API
app = AgentToBlackSheepApplicationAdapter().adapt(your_agent)

# That's it! Full API with Scalar docs at /docs
```

<img width="100%" alt="API Documentation" src="https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6" />

### 📊 Enterprise-Grade Observability

Monitor every aspect of your agents in production:

<img width="100%" alt="Observability Dashboard" src="https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7" />

**Automatic Scoring System** tracks:
- 🎯 **Model Performance** - Capability tier scoring
- 🔧 **Tool Usage** - Effectiveness metrics  
- 💰 **Cost Efficiency** - Token usage optimization
- ⚡ **Response Latency** - Performance monitoring

## 🏗️ Real-World Examples

### 💬 Customer Support Agent

```python
support_agent = Agent(
    name="Support Hero",
    instructions="You are an empathetic customer support specialist.",
    tools=[
        search_knowledge_base,
        create_ticket,
        escalate_to_human
    ],
    static_knowledge=["support_policies.pdf", "faq.md"]
)

# Deploy as API
api = AgentToBlackSheepApplicationAdapter().adapt(support_agent)
```

### 📊 Data Analysis Pipeline

```python
# Create specialized agents
data_cleaner = Agent(name="Data Cleaner", ...)
statistician = Agent(name="Statistician", ...)  
visualizer = Agent(name="Visualizer", ...)

# Build analysis pipeline
analysis_pipeline = AgentPipeline(
    agents=[data_cleaner, statistician, visualizer]
)

# Process data
result = analysis_pipeline.run(sales_dataframe)
```

### 🌍 Multi-Provider Resilience

```python
from agentle.providers import FailoverGenerationProvider

# Never go down - automatically failover between providers
resilient_provider = FailoverGenerationProvider(
    generation_providers=[
        GoogleGenaiGenerationProvider(),
        OpenAIGenerationProvider(),
        CerebrasGenerationProvider()
    ],
    shuffle=True  # Load balance
)

agent = Agent(
    generation_provider=resilient_provider,
    ...
)
```

## 🛠️ Advanced Features

### 🎭 Flexible Input Types

Agentle agents handle any input type seamlessly:

```python
# Text input
agent.run("Analyze this text")

# DataFrames
agent.run(pandas_dataframe)

# Images
agent.run(PIL_image)

# JSON/Dicts
agent.run({"user": "Alice", "query": "Help"})

# Files
agent.run(Path("document.pdf"))
```

### 🧩 Composable Prompts

Advanced templating with compile-time variable substitution:

```python
from agentle.prompts import Prompt, FSPromptProvider

# Load prompts from files
provider = FSPromptProvider(base_path="./prompts")
template = provider.provide("customer_email")

# Compile with context
email = template.compile(
    customer_name="Alice",
    issue="password reset",
    status="resolved"
)
```

### 🌊 Streaming Support

Real-time streaming for responsive experiences:

```python
async for chunk in agent.stream("Tell me a story"):
    print(chunk.text, end="")
```

## 📚 Documentation

<table>
<tr>
<td width="33%" align="center">

### 📖 [Getting Started](docs/getting-started.md)
Installation, setup, and your first agent

</td>
<td width="33%" align="center">

### 🔧 [API Reference](docs/api-reference.md)
Complete API documentation

</td>
<td width="33%" align="center">

### 🎓 [Examples](examples/)
Real-world examples and tutorials

</td>
</tr>
</table>

## 🤝 Community

<div align="center">

[![Discord](https://img.shields.io/discord/123456789?color=7289da&logo=discord&logoColor=white)](https://discord.gg/agentle)
[![Twitter Follow](https://img.shields.io/twitter/follow/agentle?style=social)](https://twitter.com/agentle)
[![GitHub Discussions](https://img.shields.io/github/discussions/yourusername/agentle)](https://github.com/yourusername/agentle/discussions)

</div>

### Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Support

- 💬 [Discord Community](https://discord.gg/agentle)
- 📧 [Email Support](mailto:support@agentle.ai)
- 🐛 [Issue Tracker](https://github.com/yourusername/agentle/issues)

## 🧠 Philosophy

> **"Simplicity is the ultimate sophistication"** - Leonardo da Vinci

Agentle was born from frustration with overly complex agent frameworks. We believe:

- **Clean APIs > Feature Creep** - Every feature must justify its complexity
- **Type Safety > Magic** - Explicit is better than implicit
- **Composability > Monoliths** - Small, focused components that work together
- **Developer Joy > Everything** - If it's not fun to use, we've failed

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>
    <strong>Built with ❤️ by developers, for developers</strong>
  </p>
  <p>
    <a href="#-agentle">⬆ Back to top</a>
  </p>
</div>