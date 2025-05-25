"""
Tool suspension error for Human-in-the-Loop workflows.

This module provides the ToolSuspensionError exception that tools can raise
to suspend agent execution and wait for external input (e.g., human approval).
"""

from typing import Any, Dict


class ToolSuspensionError(Exception):
    """
    Exception raised by tools to suspend agent execution for external input.

    This exception is used in Human-in-the-Loop workflows where a tool needs
    to pause execution and wait for human approval or other external input
    before continuing.

    When this exception is raised, the agent will:
    1. Pause execution and save the current context
    2. Return a suspended AgentRunOutput with resumption information
    3. Allow the execution to be resumed later with external input

    Attributes:
        reason: Human-readable reason for the suspension
        approval_data: Data needed for the approval process
        resumption_data: Additional data needed to resume execution
        timeout_seconds: Optional timeout for the suspension

    Example:
        ```python
        def sensitive_operation(amount: float, account: str) -> str:
            # Check if approval is needed
            if amount > 10000:
                raise ToolSuspensionError(
                    reason=f"Transfer of ${amount} requires human approval",
                    approval_data={
                        "operation": "transfer",
                        "amount": amount,
                        "account": account,
                        "risk_level": "high"
                    },
                    timeout_seconds=86400  # 24 hours
                )

            # Continue with operation if no approval needed
            return f"Transferred ${amount} to {account}"
        ```
    """

    reason: str
    approval_data: Dict[str, Any] | None = None
    resumption_data: Dict[str, Any] | None = None
    timeout_seconds: int | None = None

    def __init__(
        self,
        reason: str,
        approval_data: Dict[str, Any] | None = None,
        resumption_data: Dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ):
        """
        Initialize a tool suspension error.

        Args:
            reason: Human-readable reason for the suspension
            approval_data: Data needed for the approval process
            resumption_data: Additional data needed to resume execution
            timeout_seconds: Optional timeout for the suspension
        """
        super().__init__(reason)
        self.reason = reason
        self.approval_data = approval_data or {}
        self.resumption_data = resumption_data or {}
        self.timeout_seconds = timeout_seconds

    def __str__(self) -> str:
        return f"Tool execution suspended: {self.reason}"

    def __repr__(self) -> str:
        return (
            f"ToolSuspensionError(reason={self.reason!r}, "
            f"approval_data={self.approval_data!r}, "
            f"resumption_data={self.resumption_data!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )
