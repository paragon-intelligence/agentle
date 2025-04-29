"""
Chain of Thought reasoning model components for the Agentle framework.

This module provides structured models for representing chain-of-thought reasoning
processes in AI generations. It enables capturing step-by-step reasoning paths with
detailed explanations, supporting more transparent and explainable AI outputs.

The module includes:
- ChainOfThought: Main model representing a full reasoning process with steps and final answer
- Step: Individual reasoning step within a chain of thought
- ThoughtDetail: Granular explanation of a specific aspect within a step

These models are particularly useful for implementing reasoning strategies like
Chain of Thought (CoT), Tree of Thoughts (ToT), and other structured reasoning
approaches that make the model's thinking process explicit and verifiable.
"""

from .chain_of_thought import ChainOfThought
from .step import Step
from .thought_detail import ThoughtDetail

__all__ = ["ChainOfThought", "Step", "ThoughtDetail"]
