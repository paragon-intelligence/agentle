"""
Agent-to-Agent (A2A) Interface

This package provides a standard protocol for agent interactions in the Agentle framework.
It implements a task-based communication model where agents can exchange messages,
artifacts, and structured data using a consistent interface.

The A2A protocol enables:
- Standardized agent communication
- Session-based conversations
- Task management
- Notification systems for asynchronous updates
- Structured message and response formats

Example:
    ```python
    from agentle.agents.agent import Agent
    from agentle.agents.a2a.a2a_interface import A2AInterface
    from agentle.agents.a2a.tasks.managment.task_manager import TaskManager

    # Create an agent and task manager
    agent = Agent(...)
    task_manager = TaskManager()

    # Initialize the A2A interface
    a2a = A2AInterface(agent=agent, task_manager=task_manager)

    # Use the interface to manage tasks
    task = a2a.tasks.send(task_params)
    ```
"""

from agentle.agents.a2a.a2a_interface import A2AInterface

__all__ = ["A2AInterface"]
