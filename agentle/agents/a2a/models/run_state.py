"""
A2A Run State Model

This module defines the RunState class, which tracks the state of an agent's execution
in the A2A protocol. It maintains information about iterations, tool calls, responses,
and token usage throughout the agent's operation.
"""

from __future__ import annotations

from collections.abc import MutableSequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.generation.usage import Usage


class RunState[T_Schema = str](BaseModel):
    """
    Tracks the state of an agent's execution.

    This class maintains information about an agent's execution state, including
    iteration count, tool calls, last response, and token usage statistics. It helps
    monitor and control the execution flow of agents in the A2A protocol.

    Attributes:
        iteration: Current iteration number of the agent's execution
        tool_calls_amount: Number of tool calls made by the agent
        last_response: The most recent response from the agent
        token_usages: List of token usage statistics for each iteration

    Type Parameters:
        T_Schema: The type of the last_response, defaults to str

    Example:
        ```python
        from agentle.agents.a2a.models.run_state import RunState
        from agentle.generations.models.generation.usage import Usage

        # Initialize a new run state
        state = RunState.init_state()

        # Update the state after an iteration
        usage = Usage(input_tokens=100, output_tokens=150)
        state.update(
            last_response="Hello, I'm an AI assistant. How can I help you?",
            tool_calls_amount=0,
            iteration=1,
            token_usage=usage
        )

        # Access state information
        print(f"Current iteration: {state.iteration}")
        print(f"Tool calls made: {state.tool_calls_amount}")
        print(f"Last response: {state.last_response}")

        # Calculate total token usage
        total_input = sum(usage.input_tokens for usage in state.token_usages)
        total_output = sum(usage.output_tokens for usage in state.token_usages)
        print(f"Total tokens used: {total_input + total_output}")
        ```
    """

    iteration: int
    """Current iteration number of the agent's execution"""

    tool_calls_amount: int
    """Number of tool calls made by the agent"""

    last_response: T_Schema | str | None = None
    """The most recent response from the agent"""

    token_usages: MutableSequence[Usage] = Field(default_factory=list)
    """List of token usage statistics for each iteration"""

    @classmethod
    def init_state(cls) -> RunState[T_Schema]:
        """
        Initializes a new run state with default values.

        This class method creates a new RunState object with the iteration and
        tool calls amount set to 0, last response set to None, and an empty
        list of token usages.

        Returns:
            RunState: A new RunState object with default values

        Example:
            ```python
            from agentle.agents.a2a.models.run_state import RunState

            # Initialize a new run state
            state = RunState.init_state()

            print(f"Iteration: {state.iteration}")  # 0
            print(f"Tool calls: {state.tool_calls_amount}")  # 0
            print(f"Has response: {state.last_response is not None}")  # False
            ```
        """
        return cls(
            iteration=0,
            tool_calls_amount=0,
            last_response=None,
            token_usages=[],
        )

    def update(
        self,
        last_response: T_Schema | str,
        tool_calls_amount: int,
        iteration: int,
        token_usage: Usage,
    ) -> None:
        """
        Updates the run state with new information.

        This method updates the run state after an agent iteration, recording
        the new response, tool calls amount, iteration number, and token usage.

        Args:
            last_response: The most recent response from the agent
            tool_calls_amount: The current number of tool calls made by the agent
            iteration: The current iteration number
            token_usage: Token usage statistics for the current iteration

        Example:
            ```python
            from agentle.agents.a2a.models.run_state import RunState
            from agentle.generations.models.generation.usage import Usage

            # Initialize a run state
            state = RunState.init_state()

            # Update after first iteration
            usage1 = Usage(input_tokens=100, output_tokens=150)
            state.update(
                last_response="Hello, I'm an AI assistant.",
                tool_calls_amount=0,
                iteration=1,
                token_usage=usage1
            )

            # Update after second iteration
            usage2 = Usage(input_tokens=50, output_tokens=200)
            state.update(
                last_response="I've found the information you requested.",
                tool_calls_amount=1,
                iteration=2,
                token_usage=usage2
            )

            # Check the current state
            print(f"Current iteration: {state.iteration}")  # 2
            print(f"Total token usages recorded: {len(state.token_usages)}")  # 2
            ```
        """
        self.last_response = last_response
        self.tool_calls_amount = tool_calls_amount
        self.iteration = iteration
        self.token_usages.append(token_usage)
