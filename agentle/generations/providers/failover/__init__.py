"""
Failover provider implementation for fault-tolerant AI generation in the Agentle framework.

This module provides a fault-tolerant generation provider that can attempt to use multiple
underlying providers in sequence until one succeeds. This enables resilient applications
that can continue to function even when a primary AI provider experiences an outage or error.

The failover system can be configured to:
- Try providers in a specific order (prioritizing preferred providers)
- Randomly shuffle providers for load balancing
- Automatically capture and handle provider-specific exceptions

This implementation is particularly useful for mission-critical applications that require
high availability and cannot afford downtime due to provider-specific issues.
"""

from .failover_generation_provider import FailoverGenerationProvider

__all__: list[str] = ["FailoverGenerationProvider"]
