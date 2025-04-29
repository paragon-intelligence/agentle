"""
Message Role Module for MCP.

This module defines the Role type which represents the sender role of a message
in the Model Control Protocol (MCP) system. It specifies whether a message
is from a user or an assistant.
"""

from typing import Literal

Role = Literal["user", "assistant"]
"""
Type alias representing the role of a message sender in the MCP system.

Valid values are:
- "user": Message is from the user
- "assistant": Message is from the AI assistant
"""
