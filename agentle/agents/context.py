"""
Context management module for Agentle agents.

This module provides the Context class, which serves as a container for all contextual
information needed during an agent's execution. Context represents the conversational
state and execution history that an agent uses to generate appropriate responses.

The context includes both the message history (the conversation between user and agent)
and execution steps (a record of actions taken by the agent during processing).

Example:
```python
from agentle.agents.context import Context
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.message_parts.text import TextPart

# Create a simple conversation context
context = Context(
    message_history=[
        UserMessage(parts=[TextPart(text="Hello, can you help me with weather information?")]),
        AssistantMessage(parts=[TextPart(text="Of course! What location would you like weather for?")])
    ]
)

# Context can be passed directly to an agent
response = agent.run(context)

# Or extended with new messages
context.add_user_message("What's the weather in New York?")
```
"""

import uuid
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.execution_state import ExecutionState
from agentle.agents.step import Step
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage


class Context(BaseModel):
    """
    Container for contextual information that guides an agent's behavior.

    The Context class serves as the memory of an agent, maintaining the
    conversation history and execution record. It holds two main types of data:

    1. Messages: The sequence of communication between the user and agent,
       representing the conversation history.

    2. Steps: A record of actions taken by the agent during execution,
       such as tool calls and intermediate reasoning.

    Context objects can be:
    - Created from scratch with initial messages
    - Passed directly to an agent's run method
    - Updated with new messages during multi-turn conversations
    - Examined after execution to understand the agent's reasoning process
    - Paused and resumed for asynchronous operations

    Attributes:
        context_id: Unique identifier for this context
        message_history: The sequence of messages exchanged between the user and agent
        steps: A record of actions and execution steps taken by the agent
        execution_state: Current execution state and timing information
        metadata: Additional metadata for this context
        total_token_usage: Cumulative token usage across all operations
        session_id: Optional session identifier for grouping related contexts
        parent_context_id: Optional reference to a parent context
        tags: Optional tags for categorizing or filtering contexts

    Example:
        ```python
        # Creating a context with initial messages
        from agentle.generations.models.messages.developer_message import DeveloperMessage
        from agentle.generations.models.messages.user_message import UserMessage
        from agentle.generations.models.message_parts.text import TextPart

        context = Context(
            message_history=[
                DeveloperMessage(parts=[TextPart(text="You are a helpful weather assistant.")]),
                UserMessage(parts=[TextPart(text="What's the weather like in London?")])
            ]
        )

        # Using the context with an agent
        result = agent.run(context)

        # Examining the final context after execution
        final_context = result.context
        for message in final_context.message_history:
            print(f"{message.__class__.__name__}: {message.parts[0].text}")

        # Continuing a conversation with the same context
        final_context.add_user_message("And what about tomorrow?")
        next_result = agent.run(final_context)

        # Pausing and resuming execution
        context.pause_execution("Waiting for user input")
        # ... later ...
        context.resume_execution()
        ```
    """

    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this context."""

    message_history: MutableSequence[
        DeveloperMessage | UserMessage | AssistantMessage
    ] = Field(default_factory=list)
    """
    The sequence of messages exchanged between the user and the agent.
    Represents the full conversation history, including instructions, user queries, and agent responses.
    """

    steps: MutableSequence[Step] = Field(default_factory=list)
    """
    A record of actions and execution steps taken by the agent during processing.
    Empty by default, and populated during agent execution.
    """

    execution_state: ExecutionState = Field(default_factory=ExecutionState)
    """Current execution state and timing information."""

    metadata: MutableMapping[str, Any] = Field(default_factory=dict)
    """Additional metadata for this context."""

    total_token_usage: Usage | None = Field(default=None)
    """Cumulative token usage across all operations in this context."""

    session_id: str | None = Field(default=None)
    """Optional session identifier for grouping related contexts."""

    parent_context_id: str | None = Field(default=None)
    """Optional reference to a parent context."""

    tags: MutableSequence[str] = Field(default_factory=list)
    """Optional tags for categorizing or filtering contexts."""

    def add_user_message(self, text: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            text: The text content of the user message
        """
        message = UserMessage(parts=[TextPart(text=text)])
        self.message_history.append(message)
        self.execution_state.last_updated_at = datetime.now()

    def add_assistant_message(self, text: str) -> None:
        """
        Add an assistant message to the conversation history.

        Args:
            text: The text content of the assistant message
        """
        message = AssistantMessage(parts=[TextPart(text=text)])
        self.message_history.append(message)
        self.execution_state.last_updated_at = datetime.now()

    def add_developer_message(self, text: str) -> None:
        """
        Add a developer message to the conversation history.

        Args:
            text: The text content of the developer message
        """
        message = DeveloperMessage(parts=[TextPart(text=text)])
        self.message_history.append(message)
        self.execution_state.last_updated_at = datetime.now()

    def add_step(self, step: Step) -> None:
        """
        Add a step to the execution history.

        Args:
            step: The step to add
        """
        self.steps.append(step)
        self.execution_state.last_updated_at = datetime.now()

        # Update execution state based on step
        if step.iteration > self.execution_state.current_iteration:
            self.execution_state.current_iteration = step.iteration

        if step.has_tool_executions:
            self.execution_state.total_tool_calls += len(
                step.tool_execution_suggestions
            )

    def start_execution(self) -> None:
        """Mark the context as starting execution."""
        self.execution_state.state = "running"
        self.execution_state.started_at = datetime.now()
        self.execution_state.last_updated_at = datetime.now()

    def pause_execution(self, reason: str | None = None) -> None:
        """
        Pause the execution of this context.

        Args:
            reason: Optional reason for pausing
        """
        self.execution_state.state = "paused"
        self.execution_state.paused_at = datetime.now()
        self.execution_state.last_updated_at = datetime.now()
        if reason:
            self.execution_state.pause_reason = reason

    def resume_execution(self) -> None:
        """Resume execution of this context."""
        if self.execution_state.state == "paused" and self.execution_state.resumable:
            self.execution_state.state = "running"
            self.execution_state.paused_at = None
            self.execution_state.pause_reason = None
            self.execution_state.last_updated_at = datetime.now()

    def complete_execution(self, duration_ms: float | None = None) -> None:
        """
        Mark the context as completed.

        Args:
            duration_ms: Total execution time in milliseconds
        """
        self.execution_state.state = "completed"
        self.execution_state.completed_at = datetime.now()
        self.execution_state.last_updated_at = datetime.now()
        if duration_ms is not None:
            self.execution_state.total_duration_ms = duration_ms

    def fail_execution(
        self, error_message: str, duration_ms: float | None = None
    ) -> None:
        """
        Mark the context as failed.

        Args:
            error_message: The error message describing the failure
            duration_ms: Total execution time before failure in milliseconds
        """
        self.execution_state.state = "failed"
        self.execution_state.error_message = error_message
        self.execution_state.completed_at = datetime.now()
        self.execution_state.last_updated_at = datetime.now()
        if duration_ms is not None:
            self.execution_state.total_duration_ms = duration_ms

    def cancel_execution(self) -> None:
        """Cancel the execution of this context."""
        self.execution_state.state = "cancelled"
        self.execution_state.completed_at = datetime.now()
        self.execution_state.last_updated_at = datetime.now()

    def update_token_usage(self, usage: Usage) -> None:
        """
        Update the total token usage for this context.

        Args:
            usage: Token usage to add to the total
        """
        if self.total_token_usage is None:
            self.total_token_usage = usage
        else:
            # Add the usage to existing total
            self.total_token_usage = Usage(
                prompt_tokens=self.total_token_usage.prompt_tokens
                + usage.prompt_tokens,
                completion_tokens=self.total_token_usage.completion_tokens
                + usage.completion_tokens,
            )
        self.execution_state.last_updated_at = datetime.now()

    def set_checkpoint_data(self, key: str, value: Any) -> None:
        """
        Set checkpoint data for resumption.

        Args:
            key: The key for the checkpoint data
            value: The value to store
        """
        self.execution_state.checkpoint_data[key] = value
        self.execution_state.last_updated_at = datetime.now()

    def get_checkpoint_data(self, key: str, default: Any = None) -> Any:
        """
        Get checkpoint data for resumption.

        Args:
            key: The key for the checkpoint data
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        return self.execution_state.checkpoint_data.get(key, default)

    def clone(self, *, new_context_id: bool = True) -> "Context":
        """
        Create a copy of this context.

        Args:
            new_context_id: Whether to generate a new context ID

        Returns:
            A new Context instance with copied data
        """
        new_context = Context(
            context_id=str(uuid.uuid4()) if new_context_id else self.context_id,
            message_history=list(self.message_history),
            steps=list(self.steps),
            execution_state=self.execution_state.model_copy(deep=True),
            metadata=dict(self.metadata),
            total_token_usage=self.total_token_usage.model_copy()
            if self.total_token_usage
            else None,
            session_id=self.session_id,
            parent_context_id=self.context_id
            if new_context_id
            else self.parent_context_id,
            tags=list(self.tags),
        )
        return new_context

    @property
    def is_running(self) -> bool:
        """Check if the context is currently running."""
        return self.execution_state.state == "running"

    @property
    def is_paused(self) -> bool:
        """Check if the context is paused."""
        return self.execution_state.state == "paused"

    @property
    def is_completed(self) -> bool:
        """Check if the context has completed execution."""
        return self.execution_state.state in ["completed", "failed", "cancelled"]

    @property
    def can_resume(self) -> bool:
        """Check if the context can be resumed."""
        return self.execution_state.state == "paused" and self.execution_state.resumable

    @property
    def last_user_message(self) -> UserMessage | None:
        """Get the last user message in the conversation."""
        for message in reversed(self.message_history):
            if isinstance(message, UserMessage):
                return message
        return None

    @property
    def last_assistant_message(self) -> AssistantMessage | None:
        """Get the last assistant message in the conversation."""
        for message in reversed(self.message_history):
            if isinstance(message, AssistantMessage):
                return message
        return None
