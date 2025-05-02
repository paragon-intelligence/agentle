"""
A2A In-Memory Task Manager

This module provides an in-memory implementation of the TaskManager interface.
The InMemoryTaskManager stores tasks in a simple in-memory list, which makes it
suitable for testing, development, and simple applications that don't require
persistent storage.
"""

from typing import ClassVar

from rsb.models.field import Field

from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task import Task


class InMemoryTaskManager(TaskManager):
    """
    In-memory implementation of the TaskManager interface.

    This task manager stores tasks in a simple in-memory list, making it suitable for
    testing, development, and simple applications that don't require persistent storage.
    Tasks are lost when the application is restarted.

    Attributes:
        tasks: Class variable storing the list of tasks

    Example:
        ```python
        from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.agent import Agent

        # Create an agent and task manager
        agent = Agent(...)
        task_manager = InMemoryTaskManager()

        # Send a task
        task_params = TaskSendParams(...)
        task = task_manager.send(task_params, agent=agent)

        # The task is stored in memory
        print(f"Number of tasks in memory: {len(InMemoryTaskManager.tasks)}")
        ```

    Note:
        Since the tasks list is a class variable, all instances of InMemoryTaskManager
        share the same task storage. This means that tasks created by one instance are
        accessible by all other instances.
    """

    tasks: ClassVar[list[Task]] = Field(default_factory=list)
    """Class variable storing the list of tasks"""
