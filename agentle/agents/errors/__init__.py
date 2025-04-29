"""
Error classes for the Agentle agents framework.

This package provides exception classes used throughout the Agentle agents framework
for handling and reporting various error conditions that can occur during agent execution.

The main error types include:

- MaxToolCallsExceededError: Raised when an agent exceeds its configured maximum number
  of tool calls, preventing potential infinite loops or excessive resource consumption.

These errors are designed to provide meaningful feedback and facilitate proper
error handling in applications using the Agentle framework.

Example:
```python
from agentle.agents.errors import MaxToolCallsExceededError
from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create an agent with conservative tool call limits
agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a research assistant that can answer questions.",
    config=AgentConfig(maxToolCalls=5),  # Limit to 5 tool calls
)

try:
    result = agent.run("Research quantum computing, its history, applications, and future prospects.")
except MaxToolCallsExceededError as e:
    print(f"Agent reached tool call limit: {e}")
    # Handle the error by:
    # 1. Breaking the task into smaller parts
    # 2. Increasing the limit for this specific query
    # 3. Providing more specific instructions to make the task more efficient
```
"""

from agentle.agents.errors.max_tool_calls_exceeded_error import (
    MaxToolCallsExceededError,
)

__all__ = ["MaxToolCallsExceededError"]
