"""
A2A JSON-RPC Error Model

This module defines the JSONRPCError class, which represents error information in JSON-RPC
responses within the A2A protocol. It provides structured error reporting for API calls.
"""

from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class JSONRPCError(BaseModel):
    """
    Represents error information in JSON-RPC responses.

    This class provides a standardized structure for reporting errors in JSON-RPC
    responses within the A2A protocol. It includes an error code, message, and optional
    additional data for more detailed error information.

    Attributes:
        code: Numeric error code indicating the type of error
        message: Human-readable error message
        data: Optional additional data for more detailed error information

    Example:
        ```python
        from agentle.agents.a2a.models.json_rpc_error import JSONRPCError

        # Create a simple error
        simple_error = JSONRPCError(
            code=-32600,
            message="Invalid Request"
        )

        # Create an error with additional data
        detailed_error = JSONRPCError(
            code=-32602,
            message="Invalid params",
            data={
                "param": "sessionId",
                "reason": "Session not found",
                "details": "The specified session ID does not exist"
            }
        )

        # Access error information
        print(f"Error {detailed_error.code}: {detailed_error.message}")
        if detailed_error.data:
            print(f"Parameter: {detailed_error.data['param']}")
            print(f"Reason: {detailed_error.data['reason']}")
        ```

    Note:
        Common error codes include:
        - -32700: Parse error
        - -32600: Invalid Request
        - -32601: Method not found
        - -32602: Invalid params
        - -32603: Internal error
        - -32000 to -32099: Server error
    """

    code: int
    """Numeric error code indicating the type of error"""

    message: str
    """Human-readable error message"""

    data: dict[str, Any] | None = Field(default=None)
    """Optional additional data for more detailed error information"""
