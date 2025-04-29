"""
Adapter components for the Google AI provider integration in the Agentle framework.

This module contains adapter classes that transform responses from Google's Generative AI API
into standardized formats used throughout the Agentle framework. These adapters serve as a
crucial part of Agentle's provider abstraction layer, allowing the framework to present a
unified interface regardless of which underlying AI provider is being used.

The adapters in this package handle various conversion tasks, including:
- Transforming Google's GenerateContentResponse objects to Agentle Generation objects
- Converting Google Content objects to Agentle message formats
- Normalizing usage statistics and metadata
- Supporting structured output parsing for type-safe responses

These adapters ensure that all provider-specific details of Google's response format
are processed and normalized to Agentle's internal representation, maintaining consistency
across different AI providers within the framework.
"""
