"""
Exception module for handling maximum tool call limits in Agentle agents.

This module provides the MaxToolCallsExceededError exception, which is raised when
an agent attempts to make more tool calls than allowed by its configuration.
This exception helps prevent infinite loops or excessive resource consumption
by enforcing reasonable bounds on agent tool usage.

The error is typically raised by the Agent class when the number of iterations
in its run_async method exceeds the maxIterations setting in the AgentConfig.

Example:
```python
from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent import Agent
from agentle.agents.errors.max_tool_calls_exceeded_error import MaxToolCallsExceededError

# Create an agent with a strict limit on iterations
agent = Agent(
    # ... other parameters ...
    config=AgentConfig(maxIterations=3)
)

try:
    result = agent.run("This is a complex task that might require many tool calls")
except MaxToolCallsExceededError as e:
    print(f"Agent exceeded tool call limit: {e}")
    # Handle the error (e.g., by simplifying the task or increasing the limit)
```
"""


class MaxToolCallsExceededError(Exception):
    """
    Exception raised when an agent exceeds its configured maximum number of tool calls.

    This exception is raised during agent execution when the number of iterations
    (typically involving tool calls) exceeds the configured maximum. This is a safety
    mechanism to prevent agents from entering infinite loops or consuming excessive
    resources when dealing with complex tasks or encountering unexpected states.

    The exception is typically raised by the Agent.run_async method when state.iteration
    exceeds the maxIterations parameter from the agent's config.

    Attributes:
        message (str): A descriptive error message explaining which limit was exceeded
            and potentially how many iterations were attempted.

    Example:
        ```python
        # Inside an agent implementation
        if state.iteration >= self.config.maxIterations:
            raise MaxToolCallsExceededError(
                f"Max tool calls exceeded after {self.config.maxIterations} iterations"
            )

        # In application code using an agent
        try:
            result = agent.run(complex_query)
        except MaxToolCallsExceededError:
            # Handle the error - for example by:
            # 1. Increasing the limit for this specific task
            # 2. Breaking the task into smaller subtasks
            # 3. Providing more context to help the agent be more efficient
            agent_with_higher_limit = agent.clone(
                new_config=AgentConfig(maxIterations=20)
            )
            result = agent_with_higher_limit.run(complex_query)
        ```
    """

    pass
