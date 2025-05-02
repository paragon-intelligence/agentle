"""
A2A JSON-RPC Response Model

This module defines the JSONRPCResponse class, which represents responses in the JSON-RPC
format within the A2A protocol. It provides a standardized structure for API responses.
"""

from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.models.json_rpc_error import JSONRPCError


class JSONRPCResponse[R_Result = dict[str, Any]](BaseModel):
    """
    Represents a response in the JSON-RPC format.

    This class provides a standardized structure for API responses in the A2A protocol,
    following the JSON-RPC specification. It includes an ID to match the request,
    and either a result object or an error object.

    Attributes:
        id: String identifier matching the original request ID
        result: Optional response result data (present if the request succeeded)
        error: Optional error information (present if the request failed)

    Example:
        ```python
        from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
        from agentle.agents.a2a.models.json_rpc_error import JSONRPCError

        # Create a successful response
        success_response = JSONRPCResponse(
            id="request-123",
            result={"status": "completed", "data": {"value": 42}}
        )

        # Create an error response
        error = JSONRPCError(
            code=-32602,
            message="Invalid params",
            data={"param": "sessionId"}
        )

        error_response = JSONRPCResponse(
            id="request-123",
            error=error
        )

        # Check if a response was successful
        if success_response.result is not None:
            print(f"Request succeeded with result: {success_response.result}")
        elif success_response.error is not None:
            print(f"Request failed with error: {success_response.error.message}")
        ```

    Type Parameters:
        R_Result: The type of the result object, defaults to dict[str, Any]

    Note:
        According to the JSON-RPC specification, a response object contains either
        a result field or an error field, but not both. If the request succeeded,
        the result field is present and the error field is null. If the request failed,
        the error field is present and the result field is null.
    """

    id: str
    """String identifier matching the original request ID"""

    result: R_Result | None = Field(default=None)
    """Optional response result data (present if the request succeeded)"""

    error: JSONRPCError | None = Field(default=None)
    """Optional error information (present if the request failed)"""
