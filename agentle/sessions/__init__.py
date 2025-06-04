"""
Session management system for Agentle.

This module provides a generic session management system that can be used
across different parts of the Agentle framework, including WhatsApp bots,
web applications, and other stateful interactions.
"""

from agentle.sessions.session_store import SessionStore
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

try:
    from agentle.sessions.redis_session_store import RedisSessionStore
except ImportError:
    # Redis is optional
    RedisSessionStore = None

__all__ = [
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "SessionManager",
]
