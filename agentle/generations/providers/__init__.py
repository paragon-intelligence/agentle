"""
Provider implementations for AI generation services in the Agentle framework.

This module defines the provider abstraction layer for Agentle, which allows the framework
to interact with various AI generation services through a unified interface. The providers
package contains implementations for different AI service platforms (e.g., OpenAI, Google,
Anthropic, etc.), each implementing the common GenerationProvider interface.

The provider abstraction enables:
- Consistent API for generating text across different AI services
- Standardized handling of messages, tools, and generation parameters
- Unified response format through the Generation model
- Easy switching between providers without changing application logic
- Consistent error handling and pricing information

Applications using Agentle can work with any supported AI provider through this
abstraction layer, allowing for flexibility and provider independence.
"""

from .base.generation_provider import GenerationProvider

__all__: list[str] = ["GenerationProvider"]
