# Placeholder for OpenRouter types
"""
Type definitions for OpenRouter API requests and responses.

This module defines TypedDicts for all OpenRouter-specific structures,
ensuring type safety throughout the provider implementation.
"""

from typing import Any, Literal, NotRequired, Required, Sequence, TypedDict


class OpenRouterImageUrl(TypedDict):
    """Image URL structure for OpenRouter messages."""

    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class OpenRouterImageUrlPart(TypedDict):
    """Image URL content part."""

    type: Literal["image_url"]
    image_url: OpenRouterImageUrl


class OpenRouterCacheControl(TypedDict):
    """Cache control for prompt caching (Anthropic-style)."""

    type: Literal["ephemeral"]


class OpenRouterTextPart(TypedDict):
    """Text content part."""

    type: Literal["text"]
    text: str
    cache_control: NotRequired[OpenRouterCacheControl]


class OpenRouterFileData(TypedDict):
    """File data structure for PDFs and other documents."""

    filename: str
    file_data: str  # URL or base64 data URL


class OpenRouterFilePart(TypedDict):
    """File content part for PDFs."""

    type: Literal["file"]
    file: OpenRouterFileData


class OpenRouterInputAudioData(TypedDict):
    """Audio data structure."""

    data: str  # base64 encoded audio
    format: Literal["wav", "mp3"]


class OpenRouterInputAudioPart(TypedDict):
    """Audio content part."""

    type: Literal["input_audio"]
    input_audio: OpenRouterInputAudioData


OpenRouterMessageContent = (
    str
    | Sequence[
        OpenRouterTextPart
        | OpenRouterImageUrlPart
        | OpenRouterFilePart
        | OpenRouterInputAudioPart
    ]
)


class OpenRouterToolCallFunction(TypedDict):
    """Function call within a tool call."""

    name: str
    arguments: str  # JSON string


class OpenRouterToolCall(TypedDict):
    """Tool call structure in assistant messages."""

    id: str
    type: Literal["function"]
    function: OpenRouterToolCallFunction


class OpenRouterSystemMessage(TypedDict):
    """System/developer message format.

    ``content`` may be a plain string or a list of text parts. The list form is
    required to attach ``cache_control`` markers for prompt caching (a flat
    string cannot carry them).
    """

    role: Literal["system"]
    content: str | Sequence[OpenRouterTextPart]


class OpenRouterUserMessage(TypedDict):
    """User message format."""

    role: Literal["user"]
    content: OpenRouterMessageContent


class OpenRouterAssistantMessage(TypedDict):
    """Assistant message format."""

    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[Sequence[OpenRouterToolCall]]
    reasoning: NotRequired[str]  # Reasoning from models that support it
    reasoning_details: NotRequired[Sequence[dict[str, Any]]]


class OpenRouterToolMessage(TypedDict):
    """Tool result message format."""

    role: Literal["tool"]
    tool_call_id: str
    content: str


OpenRouterMessage = (
    OpenRouterSystemMessage
    | OpenRouterUserMessage
    | OpenRouterAssistantMessage
    | OpenRouterToolMessage
)


class OpenRouterToolFunctionParameters(TypedDict):
    """Tool function parameter schema."""

    type: Literal["object"]
    properties: dict[str, object]
    required: NotRequired[Sequence[str]]


class OpenRouterToolFunction(TypedDict):
    """Tool function definition."""

    name: str
    description: str
    parameters: OpenRouterToolFunctionParameters


class OpenRouterTool(TypedDict):
    """Tool definition structure."""

    type: Literal["function"]
    function: OpenRouterToolFunction


class OpenRouterMaxPrice(TypedDict):
    """Maximum pricing constraints."""

    prompt: NotRequired[float]  # Price per million tokens
    completion: NotRequired[float]  # Price per million tokens
    request: NotRequired[float]  # Price per request
    image: NotRequired[float]  # Price per image


class OpenRouterProviderPreferences(TypedDict):
    """Provider routing preferences."""

    allow_fallbacks: NotRequired[bool]
    require_parameters: NotRequired[bool]
    data_collection: NotRequired[Literal["allow", "deny"]]
    zdr: NotRequired[bool]  # Zero Data Retention enforcement
    order: NotRequired[Sequence[str]]
    only: NotRequired[Sequence[str]]  # Only allow these providers
    ignore: NotRequired[Sequence[str]]  # Ignore these providers
    quantizations: NotRequired[Sequence[str]]
    sort: NotRequired[Literal["price", "throughput", "latency"]]
    max_price: NotRequired[OpenRouterMaxPrice]


class OpenRouterJsonSchema(TypedDict):
    """JSON Schema for structured outputs."""

    name: str
    strict: NotRequired[bool]
    schema: dict[str, object]


class OpenRouterResponseFormat(TypedDict):
    """Response format specification for structured outputs."""

    type: Literal["json_schema"]
    json_schema: OpenRouterJsonSchema


class OpenRouterReasoning(TypedDict, total=False):
    """Reasoning or thinking configuration."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    max_tokens: int
    exclude: bool
    enabled: bool


class OpenRouterTraceConfig(TypedDict, total=False):
    """Observability trace metadata for OpenRouter."""

    trace_id: str
    trace_name: str
    span_name: str
    generation_name: str
    parent_span_id: str
    environment: str
    release: str


class OpenRouterPdfPluginConfig(TypedDict):
    """PDF parsing plugin configuration."""

    engine: NotRequired[Literal["pdf-text", "mistral-ocr", "native"]]


class OpenRouterFileParserPlugin(TypedDict):
    """File parser plugin configuration."""

    id: Literal["file-parser"]
    pdf: NotRequired[OpenRouterPdfPluginConfig]


class OpenRouterWebSearchPlugin(TypedDict):
    """Web search plugin configuration."""

    id: Literal["web"]
    engine: NotRequired[Literal["native", "exa"]]  # Search engine to use
    max_results: NotRequired[int]  # Max number of search results (default 5)
    search_prompt: NotRequired[str]  # Custom prompt for search results


OpenRouterPlugin = OpenRouterFileParserPlugin | OpenRouterWebSearchPlugin


class OpenRouterUsage(TypedDict, total=False):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any]
    completion_tokens_details: dict[str, Any]
    cost: float
    cost_details: dict[str, Any]
    is_byok: bool
    server_tool_use: dict[str, Any]


class OpenRouterResponseMessage(TypedDict):
    """Response message from OpenRouter."""

    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[Sequence[OpenRouterToolCall]]
    reasoning: NotRequired[str]  # Reasoning from models that support it
    reasoning_details: NotRequired[Sequence[dict[str, Any]]]


class OpenRouterChoice(TypedDict):
    """Choice in the response."""

    index: int
    message: OpenRouterResponseMessage
    finish_reason: str


class OpenRouterResponse(TypedDict):
    """Complete response from OpenRouter API."""

    id: str
    provider: NotRequired[str]
    model: str
    object: Literal["chat.completion"]
    created: int
    choices: Sequence[OpenRouterChoice]
    usage: OpenRouterUsage


# Streaming response types


class OpenRouterStreamDelta(TypedDict):
    """Delta content in streaming response."""

    role: NotRequired[Literal["assistant"]]
    content: NotRequired[str]
    tool_calls: NotRequired[Sequence[OpenRouterToolCall]]
    reasoning: NotRequired[str]
    reasoning_details: NotRequired[Sequence[dict[str, Any]]]


class OpenRouterStreamChoice(TypedDict):
    """Choice in streaming response."""

    index: int
    delta: OpenRouterStreamDelta
    finish_reason: NotRequired[str | None]


class OpenRouterStreamResponse(TypedDict):
    """Streaming response chunk from OpenRouter API."""

    id: str
    provider: NotRequired[str]
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: Sequence[OpenRouterStreamChoice]
    usage: NotRequired[OpenRouterUsage]


class OpenRouterRequest(TypedDict, total=False):
    """Complete request structure for OpenRouter API.

    Note: Either 'model' (single) OR 'models' (multiple with fallbacks) must be provided.
    - Use 'model' for a single model: {"model": "openai/gpt-4o"}
    - Use 'models' for fallback routing: {"models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]}
    """

    # Model selection (one of these is required)
    model: str  # Single model
    models: Sequence[str]  # Multiple models with fallback routing

    # Required fields
    messages: Required[Sequence[OpenRouterMessage]]

    # Optional parameters
    temperature: float
    max_tokens: int
    top_p: float
    top_k: float
    frequency_penalty: float
    presence_penalty: float
    stream: bool
    tools: Sequence[OpenRouterTool]
    tool_choice: Literal["auto", "none"] | dict[str, object]
    response_format: OpenRouterResponseFormat
    provider: OpenRouterProviderPreferences
    reasoning: OpenRouterReasoning
    plugins: Sequence[OpenRouterPlugin]
    transforms: Sequence[Literal["middle-out"]]  # Context compression
    session_id: str
    trace: OpenRouterTraceConfig
    user: str
    metadata: dict[str, str]


# OpenRouter Models API types


class OpenRouterModelPricing(TypedDict):
    """Pricing information for a model."""

    prompt: float | str  # Price per million tokens
    completion: float | str  # Price per million tokens
    request: NotRequired[float | str | None]  # Price per request
    image: NotRequired[float | str | None]  # Price per image
    image_output: NotRequired[float | str | None]  # Price per output image
    audio: NotRequired[float | str | None]  # Price per audio
    input_audio_cache: NotRequired[float | str | None]  # Price per cached audio
    web_search: NotRequired[float | str | None]  # Price per web search
    internal_reasoning: NotRequired[float | str | None]  # Price per reasoning token
    input_cache_read: NotRequired[float | str | None]  # Price per cache read
    input_cache_write: NotRequired[float | str | None]  # Price per cache write
    discount: NotRequired[float | None]  # Discount multiplier


class OpenRouterModelArchitecture(TypedDict):
    """Model architecture information."""

    modality: NotRequired[str]
    tokenizer: NotRequired[str]
    instruct_type: NotRequired[str | None]


class OpenRouterModelTopProvider(TypedDict):
    """Information about the top provider."""

    context_length: NotRequired[int | None]
    max_completion_tokens: NotRequired[int | None]
    is_moderated: NotRequired[bool]


class OpenRouterModelPerRequestLimits(TypedDict):
    """Per-request token limits."""

    prompt_tokens: NotRequired[int | str | None]
    completion_tokens: NotRequired[int | str | None]


class OpenRouterModelDefaultParams(TypedDict):
    """Default parameters for the model."""

    temperature: NotRequired[float]
    top_p: NotRequired[float]
    top_k: NotRequired[int]


class OpenRouterModel(TypedDict):
    """Model information from OpenRouter /models endpoint."""

    id: str
    canonical_slug: NotRequired[str]
    name: str
    created: NotRequired[float]
    pricing: OpenRouterModelPricing
    context_length: NotRequired[int | None]
    architecture: NotRequired[OpenRouterModelArchitecture]
    top_provider: NotRequired[OpenRouterModelTopProvider]
    per_request_limits: NotRequired[OpenRouterModelPerRequestLimits | None]
    supported_parameters: NotRequired[Sequence[str]]
    default_parameters: NotRequired[OpenRouterModelDefaultParams]
    hugging_face_id: NotRequired[str | None]
    description: NotRequired[str | None]


class OpenRouterModelsResponse(TypedDict):
    """Response from OpenRouter /models endpoint."""

    data: Sequence[OpenRouterModel]
