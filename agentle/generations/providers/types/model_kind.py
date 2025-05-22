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
