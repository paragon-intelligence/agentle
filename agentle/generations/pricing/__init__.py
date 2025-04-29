"""
Pricing module for the Agentle generations package.

This module provides utilities and classes for calculating, tracking, and managing costs
associated with language model generations. It implements a flexible pricing system that
can accommodate different pricing strategies and provider-specific pricing models.

Key components:

- PriceRetrievable: An abstract base class for objects that can retrieve pricing information
  for language models.
- PricingCalculationStrategy: An abstract base class defining the interface for different
  pricing calculation strategies.
- DefaultInternalLLMPriceCalculationStrategy: A standard implementation of the pricing calculation
  strategy that computes costs based on input and output tokens.

The pricing module enables accurate cost calculation for AI model usage based on token counts,
which is essential for budgeting, monitoring, and optimizing AI application costs.

Example:
```python
from agentle.generations.pricing.default_internal_llm_price_calculator import DefaultInternalLLMPriceCalculationStrategy
from agentle.generations.providers.some_provider import SomeProvider

# Create a pricing calculator
calculator = DefaultInternalLLMPriceCalculationStrategy()

# Calculate price for a generation
provider = SomeProvider()
price = calculator.calculate_price(
    input_tokens=1000,
    completion_tokens=500,
    model="model-name",
    provider=provider
)

print(f"Generation cost: ${price:.6f}")
```
"""

# Import public modules and classes
from agentle.generations.pricing.default_internal_llm_price_calculator import (
    DefaultInternalLLMPriceCalculationStrategy,
)
from agentle.generations.pricing.price_retrievable import PriceRetrievable
from agentle.generations.pricing.pricing_calculator import PricingCalculationStrategy

__all__ = [
    "DefaultInternalLLMPriceCalculationStrategy",
    "PriceRetrievable",
    "PricingCalculationStrategy",
]
