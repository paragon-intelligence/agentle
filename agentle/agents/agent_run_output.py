"""
Module for representing and managing agent execution results.

This module provides the AgentRunOutput class which encapsulates all data
produced during an agent's execution cycle. It represents both the final response
and metadata about the execution process, including resource usage statistics,
conversation context, and structured outputs.

Example:
```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create and run an agent
agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant."
)

# The result is an AgentRunOutput object
result = agent.run("What is the capital of France?")

# Access different aspects of the result
text_response = result.artifacts[0].parts[0].text
resource_usage = result.usage
conversation_context = result.final_context
structured_data = result.parsed  # If using a response_schema
```
"""

import logging
from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.context import Context
from agentle.agents.models.agent_usage_statistics import AgentUsageStatistics
from agentle.agents.models.artifact import Artifact
from agentle.agents.tasks.task_state import TaskState

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    """
    Represents the complete result of an agent execution.

    AgentRunOutput encapsulates all data produced when an agent is run, including
    the primary response artifacts, execution metadata, resource usage statistics,
    and optionally structured output data when a response schema is provided.

    This class is generic over T_StructuredOutput, which represents the optional
    structured data format that can be extracted from the agent's response when
    a response schema is specified.

    Attributes:
        artifacts (Sequence[Artifact]): The primary outputs produced by the agent,
            typically containing text, images, or other content in response to
            the user's input. Most commonly, the first artifact contains the main
            response text.

        model_name (str): The name of the model that was used to generate the response,
            useful for tracking and debugging purposes.

        task_status (TaskState): The final status of the agent execution task,
            indicating whether it completed successfully, was interrupted, etc.

        usage (AgentUsageStatistics): Statistics about resource utilization during
            execution, such as token counts, which is valuable for monitoring,
            billing, and optimizing resource usage.

        final_context (Context): The complete conversation context at the end of
            execution, including all messages exchanged during the agent run.
            This is useful for maintaining conversation state across multiple
            interactions.

        parsed (T_StructuredOutput): The structured data extracted from the agent's
            response when a response schema was provided. This will be None if
            no schema was specified. When present, it contains a strongly-typed
            representation of the agent's output, conforming to the specified schema.

    Example:
        ```python
        # Basic usage to access the text response
        result = agent.run("Tell me about Paris")
        response_text = result.artifacts[0].parts[0].text
        print(response_text)

        # Checking task completion status
        if result.task_status == TaskState.COMPLETED:
            print("Task completed successfully")

        # Working with structured output
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            name: str
            country: str
            population: int

        structured_agent = Agent(
            # ... other parameters ...
            response_schema=CityInfo
        )

        result = structured_agent.run("Tell me about Paris")
        if result.parsed:
            print(f"{result.parsed.name} is in {result.parsed.country}")
            print(f"Population: {result.parsed.population}")
        ```
    """

    artifacts: Sequence[Artifact]
    """
    The primary outputs produced by the agent in response to the user's input.
    Typically contains text, images, or other content.
    """

    model_name: str
    """
    The name of the model that was used to generate the response.
    """

    task_status: TaskState
    """
    The final status of the agent execution task (completed, failed, etc.).
    """

    usage: AgentUsageStatistics
    """
    Statistics about resource utilization during execution, such as token counts.
    """

    final_context: Context
    """
    The complete conversation context at the end of execution.
    """

    parsed: T_StructuredOutput
    """
    Structured data extracted from the agent's response when a response schema was provided.
    Will be None if no schema was specified.
    """
