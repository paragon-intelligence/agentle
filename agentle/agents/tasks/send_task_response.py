from rsb.models.field import Field

from agentle.agents.models.json_rpc_error import JSONRPCError
from agentle.agents.models.json_rpc_response import JSONRPCResponse
from agentle.agents.tasks.task import Task


class SendTaskResponse(JSONRPCResponse[Task]):
    """
    Response to a send task request.
    """

    result: Task | None = Field(default=None, description="Task result")
    error: JSONRPCError | None = Field(
        default=None, description="Error if the request failed"
    )
