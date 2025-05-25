"""
Tools package for Agentle framework.

This package provides tools functionality for the Agentle framework, allowing AI models
to interact with external systems and perform actions. The primary class exposed is the
Tool class, which represents a callable function with associated metadata.

Tools can be created either directly from Python functions using the `Tool.from_callable`
method or from MCP (Model Control Protocol) tools using the `Tool.from_mcp_tool` method.

Example:
```python
from agentle.generations.tools import Tool

# Create a tool from a function
def get_weather(location: str) -> str:
    \"\"\"Get the current weather for a location\"\"\"
    return f"The weather in {location} is sunny."

weather_tool = Tool.from_callable(get_weather)

# Use the tool
result = weather_tool.call(location="Tokyo")
print(result)  # "The weather in Tokyo is sunny."
```
"""

from agentle.generations.tools.tool import Tool
from agentle.generations.tools.decorators.tool import tool

__all__ = ["Tool", "tool"]
