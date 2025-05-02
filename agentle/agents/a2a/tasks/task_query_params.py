"""
A2A Task Query Parameters

This module defines the TaskQueryParams class, which represents the parameters for
querying a task in the A2A protocol. It encapsulates the information needed to retrieve
a specific task and control how much history is included in the response.
"""

from typing import Any
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TaskQueryParams(BaseModel):
    """
    Parameters for querying a task in the A2A protocol.

    This class encapsulates the information needed to retrieve a specific task
    and control how much history is included in the response.

    Attributes:
        id: The identifier of the task to retrieve
        historyLength: Optional limit on the number of history messages to include
        metadata: Optional additional metadata for the query

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
        from agentle.agents.a2a.tasks.managment.task_manager import TaskManager

        # Create a simple query to retrieve a task
        query = TaskQueryParams(id="task-123")

        # Create a query with limited history
        detailed_query = TaskQueryParams(
            id="task-123",
            historyLength=5,  # Limit to last 5 messages
            metadata={"include_artifacts": True}
        )

        # Retrieve the task result
        task_manager = TaskManager()
        result = task_manager.get(query, agent=agent)
        ```
    """

    id: str
    """The identifier of the task to retrieve"""

    historyLength: int | None = Field(default=None)
    """Optional limit on the number of history messages to include"""

    metadata: dict[str, Any] | None = Field(default=None)
    """Optional additional metadata for the query"""
