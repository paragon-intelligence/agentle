"""
MCP Sampling Package

This package provides classes and utilities for model sampling, completion requests,
and handling model responses. It defines the data structures for structuring requests
to language models and processing their outputs.

Components include:
- Request models for model sampling
- Response/completion result structures
- Model preferences and configuration
- Message content types (text, images)
"""

from agentle.mcp.sampling.completion_result import CompletionResult
from agentle.mcp.sampling.hint import Hint
from agentle.mcp.sampling.model_preferences import ModelPreference
from agentle.mcp.sampling.request import SamplingRequest

__all__ = [
    "CompletionResult",
    "Hint",
    "ModelPreference",
    "SamplingRequest",
]
