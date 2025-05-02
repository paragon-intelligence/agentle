"""
A2A Task Get Result

This module defines the TaskGetResult class, which represents the result of a task query
in the A2A protocol. It encapsulates the retrieved task and any error information.
"""

import uuid

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.tasks.task import Task


class TaskGetResult(BaseModel):
    """
    Result of a task query in the A2A protocol.

    This class encapsulates the retrieved task and any error information that might
    have occurred during the query. It provides a consistent structure for task
    retrieval results.

    Attributes:
        id: Unique identifier for the result (auto-generated if not provided)
        result: The retrieved task
        error: Optional error message if the query failed

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
        from agentle.agents.a2a.tasks.managment.task_manager import TaskManager

        # Query a task
        task_manager = TaskManager()
        query = TaskQueryParams(id="task-123")
        result = task_manager.get(query, agent=agent)

        # Check for errors
        if result.error:
            print(f"Error retrieving task: {result.error}")
        else:
            # Access the task
            task = result.result
            print(f"Task status: {task.status}")

            # Access task history
            if task.history:
                for message in task.history:
                    for part in message.parts:
                        if part.type == "text":
                            print(f"{message.role}: {part.text}")
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the result"""

    result: Task
    """The retrieved task"""

    error: str | None = Field(default=None)
    """Optional error message if the query failed"""
