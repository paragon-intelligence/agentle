"""
A2A Resources Package

This package provides resource classes for interacting with different aspects of the A2A protocol.
Resources represent logical groupings of related functionality and serve as interfaces for
clients to interact with agents, tasks, and notifications.

The package includes:
- TaskResource for managing tasks and their lifecycle
- PushNotificationResource for configuring and managing notifications

Example:
    ```python
    from agentle.agents.agent import Agent
    from agentle.agents.a2a.resources.task_resource import TaskResource
    from agentle.agents.a2a.tasks.managment.task_manager import TaskManager

    # Create an agent and task manager
    agent = Agent(...)
    task_manager = TaskManager()

    # Create a task resource
    task_resource = TaskResource(agent=agent, manager=task_manager)

    # Use the resource to send a task
    task = task_resource.send(task_params)

    # Use the resource to get task results
    result = task_resource.get(query_params={"id": task.id})

    # Access notification functionality
    notification_resource = task_resource.pushNotification
    ```
"""

from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.resources.push_notification_resource import (
    PushNotificationResource,
)

__all__ = ["TaskResource", "PushNotificationResource"]
