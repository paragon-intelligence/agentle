"""
A2A Task State

This module defines the TaskState enum, which represents the possible states of a task
in the A2A protocol. Task states are used to track the progress and status of tasks
as they move through their lifecycle.
"""

from enum import Enum


class TaskState(str, Enum):
    """
    Enum representing the possible states of a task in the A2A protocol.

    Task states are used to track the progress and status of tasks as they move
    through their lifecycle, from submission to completion (or failure).

    States:
        SUBMITTED: Task has been submitted but processing has not yet begun
        WORKING: Task is currently being processed by the agent
        INPUT_REQUIRED: Task requires additional input from the user to proceed
        COMPLETED: Task has been successfully completed
        FAILED: Task has failed to complete due to an error
        CANCELED: Task has been canceled by the user or system

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_state import TaskState
        from agentle.agents.a2a.tasks.task import Task

        # Create a task in the SUBMITTED state
        task = Task(
            sessionId="session-123",
            status=TaskState.SUBMITTED
        )

        # Check the task state
        if task.status == TaskState.SUBMITTED:
            print("Task is waiting to be processed")

        # Update the task state to WORKING
        task.status = TaskState.WORKING

        # Check if a task is complete
        is_terminal_state = task.status in [
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED
        ]
        ```
    """

    SUBMITTED = "submitted"
    """Task has been submitted but processing has not yet begun"""

    WORKING = "working"
    """Task is currently being processed by the agent"""

    INPUT_REQUIRED = "input-required"
    """Task requires additional input from the user to proceed"""

    COMPLETED = "completed"
    """Task has been successfully completed"""

    FAILED = "failed"
    """Task has failed to complete due to an error"""

    CANCELED = "canceled"
    """Task has been canceled by the user or system"""
