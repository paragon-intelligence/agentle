"""
A2A Task Manager

This module defines the TaskManager abstract base class, which provides the interface for
managing tasks in the A2A protocol. The TaskManager is responsible for handling the lifecycle
of tasks, including creation, retrieval, and notification subscription.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams

type WithoutStructuredOutput = None


class TaskManager(ABC):
    """
    Abstract base class for task management in the A2A protocol.

    This class defines the interface for creating, retrieving, and managing tasks.
    Concrete implementations, such as InMemoryTaskManager, provide the specific
    storage and execution behavior.

    Methods:
        send: Creates and starts a new task or continues an existing session
        get: Retrieves a task based on query parameters
        send_subscribe: Sends a task and sets up a subscription for updates
        cancel: Cancels an ongoing task
    """

    @abstractmethod
    async def send(
        self,
        task_params: TaskSendParams,
        agent: Any,  # Type at runtime, actual typing happens through TYPE_CHECKING
    ) -> Task:
        """
        Creates and starts a new task or continues an existing session.

        Args:
            task_params: Parameters for the task to create
            agent: The agent to execute the task

        Returns:
            Task: The created or updated task
        """
        pass

    @abstractmethod
    async def get(
        self,
        query_params: TaskQueryParams,
        agent: Any,  # Type at runtime, actual typing happens through TYPE_CHECKING
    ) -> TaskGetResult:
        """
        Retrieves a task based on query parameters.

        Args:
            query_params: Parameters to query the task
            agent: The agent associated with the task

        Returns:
            TaskGetResult: The result of the task
        """
        pass

    @abstractmethod
    async def send_subscribe(
        self,
        task_params: TaskSendParams,
        agent: Any,  # Type at runtime, actual typing happens through TYPE_CHECKING
    ) -> JSONRPCResponse:
        """
        Sends a task and sets up a subscription for updates.

        Args:
            task_params: Parameters for the task to create
            agent: The agent to execute the task

        Returns:
            JSONRPCResponse: The response containing subscription information
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """
        Cancels an ongoing task.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            bool: True if the task was successfully canceled, False otherwise
        """
        pass
