"""
A2A Send Task Response Model

This module defines the SendTaskResponse class, which represents the response to a
send task request in the A2A protocol. It follows the JSON-RPC response format and
includes either the created task or error information.
"""

from rsb.models.field import Field

from agentle.agents.a2a.models.json_rpc_error import JSONRPCError
from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.tasks.task import Task


class SendTaskResponse(JSONRPCResponse[Task]):
    """
    Represents the response to a send task request.

    This class extends JSONRPCResponse with Task as the result type. It includes
    either the created task (if successful) or error information (if failed),
    following the JSON-RPC response format.

    Attributes:
        result: The created task if the request was successful
        error: Error information if the request failed

    Example:
        ```python
        from agentle.agents.a2a.tasks.send_task_response import SendTaskResponse
        from agentle.agents.a2a.tasks.task import Task
        from agentle.agents.a2a.models.json_rpc_error import JSONRPCError
        from agentle.agents.a2a.tasks.task_state import TaskState

        # Create a successful response
        task = Task(
            sessionId="session-123",
            status=TaskState.SUBMITTED
        )

        success_response = SendTaskResponse(
            id="request-456",
            result=task
        )

        # Create an error response
        error = JSONRPCError(
            code=-32602,
            message="Invalid task parameters",
            data={"param": "message", "reason": "Message is required"}
        )

        error_response = SendTaskResponse(
            id="request-789",
            error=error
        )

        # Check if a response indicates success
        if success_response.result is not None:
            print(f"Task created with ID: {success_response.result.id}")
        elif success_response.error is not None:
            print(f"Failed to create task: {success_response.error.message}")
        ```

    Note:
        According to the JSON-RPC specification and the parent JSONRPCResponse class,
        a response contains either a result field or an error field, but not both.
        For successful requests, the result field contains the created task and the error
        field is None. For failed requests, the error field contains error information
        and the result field is None.
    """

    result: Task | None = Field(default=None, description="Task result")
    """The created task if the request was successful"""

    error: JSONRPCError | None = Field(
        default=None, description="Error if the request failed"
    )
    """Error information if the request failed"""
