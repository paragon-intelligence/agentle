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
    messages=[
        UserMessage(parts=[TextPart(text="Hello, can you help me with weather information?")]),
        AssistantMessage(parts=[TextPart(text="Of course! What location would you like weather for?")])
    ]
)

# Context can be passed directly to an agent
response = agent.run(context)

# Or extended with new messages
context.messages = list(context.messages) + [
    UserMessage(parts=[TextPart(text="What's the weather in New York?")])
]
```
"""

from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.step import Step
from agentle.generations.models.messages.message import Message


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

    Attributes:
        messages (Sequence[Message]): The sequence of messages exchanged between
            the user and the agent, representing the conversation history. This
            typically includes at minimum a system/developer message defining
            the agent's instructions, and a user message containing the query.

        steps (Sequence[Step]): A record of actions and execution steps taken
            by the agent during processing. This can include reasoning steps,
            tool calls, and other operations performed by the agent. Empty by
            default, and populated during agent execution.

    Example:
        ```python
        # Creating a context with initial messages
        from agentle.generations.models.messages.developer_message import DeveloperMessage
        from agentle.generations.models.messages.user_message import UserMessage
        from agentle.generations.models.message_parts.text import TextPart

        context = Context(
            messages=[
                DeveloperMessage(parts=[TextPart(text="You are a helpful weather assistant.")]),
                UserMessage(parts=[TextPart(text="What's the weather like in London?")])
            ]
        )

        # Using the context with an agent
        result = agent.run(context)

        # Examining the final context after execution
        final_context = result.final_context
        for message in final_context.messages:
            print(f"{message.__class__.__name__}: {message.parts[0].text}")

        # Continuing a conversation with the same context
        updated_messages = list(final_context.messages) + [
            UserMessage(parts=[TextPart(text="And what about tomorrow?")])
        ]
        new_context = Context(messages=updated_messages)
        ```
    """

    messages: Sequence[Message]
    """
    The sequence of messages exchanged between the user and the agent.
    Represents the full conversation history, including instructions, user queries, and agent responses.
    """

    steps: Sequence[Step] = Field(default_factory=list)
    """
    A record of actions and execution steps taken by the agent during processing.
    Empty by default, and populated during agent execution.
    """
