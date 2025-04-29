"""
Error Code Definitions

This module defines standard error codes used throughout the MCP implementation,
following JSON-RPC 2.0 error code conventions.
"""

from enum import Enum


class ErrorCode(Enum):
    """
    Standard JSON-RPC 2.0 error codes used in MCP communications.

    These error codes align with the JSON-RPC 2.0 specification and provide
    consistent error reporting across the MCP framework.

    Attributes:
        PARSE_ERROR (-32700): Invalid JSON received by the server.
        INVALID_REQUEST (-32600): The JSON sent is not a valid request object.
        METHOD_NOT_FOUND (-32601): The method does not exist or is unavailable.
        INVALID_PARAMS (-32602): Invalid method parameters.
        INTERNAL_ERROR (-32603): Internal JSON-RPC error.
    """

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
