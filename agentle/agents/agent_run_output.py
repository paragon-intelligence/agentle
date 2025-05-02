"""
Module for representing and managing agent execution results.

This module provides the AgentRunOutput class which encapsulates all data
produced during an agent's execution cycle. It represents both the final response
and metadata about the execution process, including conversation steps and structured outputs.

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
text_response = result.generation.text
conversation_steps = result.steps
structured_data = result.parsed  # If using a response_schema
```
"""

import logging
from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.step import Step
from agentle.generations.models.generation.generation import Generation

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    """
    Represents the complete result of an agent execution.

    AgentRunOutput encapsulates all data produced when an agent is run, including
    the primary generation response, conversation steps, and optionally
    structured output data when a response schema is provided.

    This class is generic over T_StructuredOutput, which represents the optional
    structured data format that can be extracted from the agent's response when
    a response schema is specified.

    Attributes:
        generation (Generation[T_StructuredOutput]): The primary generation produced by the agent,
            containing the response to the user's input. This includes text, potentially images,
            and any other output format supported by the model.

        steps (Sequence[Step]): The sequence of conversation steps that occurred during the
            agent execution. This includes user inputs, model responses, and any intermediate
            steps that were part of the execution flow.

        parsed (T_StructuredOutput): The structured data extracted from the agent's
            response when a response schema was provided. This will be None if
            no schema was specified. When present, it contains a strongly-typed
            representation of the agent's output, conforming to the specified schema.

    Example:
        ```python
        # Basic usage to access the text response
        result = agent.run("Tell me about Paris")
        response_text = result.generation.text
        print(response_text)

        # Examining conversation steps
        for step in result.steps:
            print(f"Step type: {step.type}")
            print(f"Content: {step.content}")

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

    generation: Generation[T_StructuredOutput]
    """
    The generation produced by the agent.
    """

    steps: Sequence[Step]
    """
    The complete conversation context at the end of execution.
    """

    parsed: T_StructuredOutput
    """
    Structured data extracted from the agent's response when a response schema was provided.
    Will be None if no schema was specified.
    """

    @property
    def text(self) -> str:
        """
        The text response from the agent.
        """
        return self.generation.text
