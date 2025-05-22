"""
Model capability categories for provider-agnostic model selection.

This module defines the ModelKind type alias, which provides abstract capability
categories that can be mapped to specific provider models. Using ModelKind values
instead of provider-specific model names enables:

1. Provider independence: Write code that works with any AI provider
2. Future-proofing: When providers release new models, only mapping tables need updates
3. Capability-based selection: Choose models based on capabilities, not names
4. Simplified failover: When using FailoverGenerationProvider, each provider
   automatically maps to its equivalent model
5. Consistency: Standardized categories across all providers

Each provider implements map_model_kind_to_provider_model() to translate these
abstract categories to their specific models (e.g., "category_pro" → "gpt-4o"
for OpenAI or "gemini-2.5-pro" for Google).

This abstraction is particularly valuable for:
- Multi-provider applications that need to work with any AI provider
- Failover scenarios where requests can seamlessly switch between providers
- Future-proofing code against model name changes and new model releases
"""

from typing import Literal, TypeAlias

ModelKind: TypeAlias = Literal[
    "category_nano",  # Smallest, fastest, most cost-effective (GPT-4.1 nano, etc.)
    "category_mini",  # Small but capable models (GPT-4.1 mini, Claude Haiku)
    "category_standard",  # Mid-range, balanced performance (Claude Sonnet, Gemini Flash)
    "category_pro",  # High performance models (Gemini Pro, etc.)
    "category_flagship",  # Best available model from provider (Claude Opus, GPT-4.5)
    "category_reasoning",  # Specialized for complex reasoning (o1, o3-mini, hybrid models)
    "category_vision",  # Multimodal capabilities for image/video processing
    "category_coding",  # Specialized for programming tasks (Claude Code-optimized models)
    "category_instruct",  # Fine-tuned for instruction following (Turbo-Instruct style),
    # Experimental variants
    "category_nano_experimental",
    "category_mini_experimental",
    "category_standard_experimental",
    "category_pro_experimental",
    "category_flagship_experimental",
    "category_reasoning_experimental",
    "category_vision_experimental",
    "category_coding_experimental",
    "category_instruct_experimental",
]
