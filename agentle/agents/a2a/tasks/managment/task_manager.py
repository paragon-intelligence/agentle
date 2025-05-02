"""
A2A Task Manager

This module defines the TaskManager abstract base class, which provides the interface for
managing tasks in the A2A protocol. The TaskManager is responsible for handling the lifecycle
of tasks, including creation, retrieval, and notification subscription.
"""

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.agent import Agent

type WithoutStructuredOutput = None


class TaskManager(BaseModel):
    """
    Abstract base class for managing tasks in the A2A protocol.

    The TaskManager is responsible for handling the lifecycle of tasks, including
    creation, retrieval, and notification subscription. It provides a consistent
    interface for different implementations (e.g., in-memory, database-backed).

    Implementations of this class must provide concrete methods for sending tasks,
    retrieving task results, and setting up subscriptions for task updates.

    Example:
        ```python
        from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
        from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
        from agentle.agents.agent import Agent

        # Create an agent and task manager
        agent = Agent(...)
        task_manager = ConcreteTaskManager()  # Some implementation of TaskManager

        # Send a task
        task_params = TaskSendParams(...)
        task = task_manager.send(task_params, agent=agent)

        # Retrieve task results
        query_params = TaskQueryParams(id=task.id)
        result = task_manager.get(query_params, agent=agent)
        ```
    """

    def send[T_Schema = WithoutStructuredOutput](
        self, task: TaskSendParams, agent: Agent[T_Schema]
    ) -> Task:
        """
        Sends a task to an agent for processing.

        This method takes task parameters and an agent, creates a new task or
        continues an existing session, and returns the resulting task.

        Args:
            task: The parameters for the task to send
            agent: The agent to process the task

        Returns:
            Task: The created or updated task

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_send_params import TaskSendParams

            # Create task parameters
            task_params = TaskSendParams(...)

            # Send the task
            task = task_manager.send(task_params, agent=agent)
            ```
        """
        ...

    def get[T_Schema = WithoutStructuredOutput](
        self, query_params: TaskQueryParams, agent: Agent[T_Schema]
    ) -> TaskGetResult:
        """
        Retrieves the result of a task.

        This method takes query parameters and an agent, retrieves the
        specified task, and returns its result.

        Args:
            query_params: The parameters for querying the task
            agent: The agent associated with the task

        Returns:
            TaskGetResult: The result of the task

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams

            # Create query parameters
            query_params = TaskQueryParams(id="task-123")

            # Retrieve the task result
            result = task_manager.get(query_params, agent=agent)
            ```
        """
        ...

    def send_subscribe[T_Schema = WithoutStructuredOutput](
        self, task: TaskSendParams, agent: Agent[T_Schema]
    ) -> JSONRPCResponse:
        """
        Sends a task and subscribes to updates.

        This method takes task parameters and an agent, creates a new task or
        continues an existing session, and sets up a subscription to receive
        updates about the task's progress.

        Args:
            task: The parameters for the task to send
            agent: The agent to process the task

        Returns:
            JSONRPCResponse: The response containing subscription information

        Example:
            ```python
            from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
            from agentle.agents.a2a.notifications.push_notification_config import PushNotificationConfig

            # Create task parameters with notification config
            notification_config = PushNotificationConfig(
                url="https://example.com/notifications",
                token="notification-token-123"
            )

            task_params = TaskSendParams(
                message=message,
                sessionId="analysis-session",
                pushNotification=notification_config
            )

            # Send the task and subscribe to updates
            response = task_manager.send_subscribe(task_params, agent=agent)
            ```
        """
        ...
