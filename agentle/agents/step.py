"""
Module for tracking execution steps in Agentle agents.

This module provides the Step class, which represents a single execution step
in an agent's reasoning process. Steps track the specific actions taken by an
agent during its execution, particularly focusing on tool calls.

Steps are stored in the Context object's 'steps' sequence and provide a
detailed record of the agent's decision-making and actions during execution.
This is useful for debugging, logging, and understanding the agent's reasoning path.

Example:
```python
from agentle.agents.step import Step
from agentle.generations.models.message_parts.tool_execution_suggestion import ToolExecutionSuggestion

# Create a tool execution suggestion
tool_suggestion = ToolExecutionSuggestion(
    id="tool-call-1",
    tool_name="get_weather",
    args={"location": "London"}
)

# Create a step recording that tool call
step = Step(called_tools=[tool_suggestion])

# Steps are typically stored in a Context object
context.steps = list(context.steps) + [step]
```
"""

from collections.abc import Sequence
from rsb.models.base_model import BaseModel

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class Step(BaseModel):
    """
    Represents a single execution step in an agent's reasoning process.

    The Step class tracks the actions and decisions made by an agent during execution,
    with a primary focus on tool calls. Each step captures a discrete point in the
    agent's execution where it interacted with external tools or made significant
    progress in its reasoning.

    Steps serve several important purposes:
    1. Tracking the sequence of tool calls made by the agent
    2. Providing a chronological record of agent actions for debugging
    3. Supporting audit trails of what external resources were accessed
    4. Facilitating analysis of agent decision-making patterns

    Steps are stored in the Context object's 'steps' sequence. They are primarily
    used internally by the agent execution logic, but are exposed in the
    AgentRunOutput for inspection and logging.

    Attributes:
        called_tools (Sequence[ToolExecutionSuggestion]): A sequence of tool execution
            suggestions that were made during this step. Each suggestion contains
            information about which tool was called, with what arguments, and its
            unique identifier.

    Example:
        ```python
        # Creating a step to record a weather API call
        from agentle.generations.models.message_parts.tool_execution_suggestion import ToolExecutionSuggestion

        weather_tool_call = ToolExecutionSuggestion(
            id="call-123",
            tool_name="get_weather",
            args={"location": "New York", "units": "celsius"}
        )

        step = Step(called_tools=[weather_tool_call])

        # Steps are typically added to a context during agent execution
        context.steps = list(context.steps) + [step]

        # Examining steps after execution
        for step in result.final_context.steps:
            for tool_call in step.called_tools:
                print(f"Called: {tool_call.tool_name} with args: {tool_call.args}")
        ```
    """

    called_tools: Sequence[ToolExecutionSuggestion]
    """
    A sequence of tool execution suggestions that were made during this step.
    
    Each ToolExecutionSuggestion contains information about a specific tool call,
    including the tool name, arguments passed to the tool, and a unique identifier
    for the call.
    """
