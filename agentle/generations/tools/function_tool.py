"""
Function tool decorator module for the Agentle framework.

This module provides a simple decorator function that converts regular Python functions
into Tool instances that can be used by AI agents in the Agentle framework.

The function_tool decorator is a convenient shorthand for Tool.from_callable, making it
easier to create tools from functions without explicitly calling the class method.

Example:
```python
from agentle.generations.tools.function_tool import function_tool

@function_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    \"\"\"Get current weather for a location\"\"\"
    # Implementation would typically call a weather API
    return f"The weather in {location} is sunny. Temperature is 25°{unit[0].upper()}"

# The function is now a Tool instance
print(get_weather.name)  # "get_weather"
print(get_weather.description)  # "Get current weather for a location"

# Call the tool with parameters
result = get_weather.call(location="Tokyo", unit="fahrenheit")
```
"""

from collections.abc import Callable

from agentle.generations.tools.tool import Tool


def function_tool(func: Callable[..., object]) -> Tool:
    """
    Decorator that converts a Python function into a Tool instance.

    This decorator wraps the provided function using Tool.from_callable,
    creating a Tool instance that can be used by AI agents in the Agentle framework.
    The resulting Tool automatically inherits the function's name, docstring as description,
    and parameter specifications based on type annotations and default values.

    Args:
        func: The callable function to convert into a Tool.

    Returns:
        Tool: A Tool instance wrapping the provided function.

    Example:
        ```python
        @function_tool
        def search_database(query: str, limit: int = 10) -> list[dict]:
            \"\"\"Search the database for records matching the query\"\"\"
            # Implementation would typically search a database
            return [{"id": 1, "result": f"Result for {query}"}] * min(limit, 100)

        # Now search_database is a Tool instance
        # Use it directly
        results = search_database.call(query="test", limit=5)

        # Or pass it to an agent as a tool
        agent = Agent(tools=[search_database])
        ```
    """
    return Tool.from_callable(func)
