import json

import httpx
import pytest

from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.openrouter.openrouter_generation_provider import (
    OpenRouterGenerationProvider,
)


def _build_provider(handler, model: str = "openai/o3-mini"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterGenerationProvider(api_key="test-key", http_client=client)
    provider._models_cache = {
        model: {
            "id": model,
            "name": model,
            "pricing": {
                "prompt": 0.0,
                "completion": 0.0,
            },
        }
    }
    return provider, client


@pytest.mark.asyncio
async def test_generate_async_sends_reasoning_config_to_openrouter():
    captured_request_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request_body
        captured_request_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/o3-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Hello from OpenRouter",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {
                        "cached_tokens": 4,
                        "cache_write_tokens": 2,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 3,
                        "image_tokens": 1,
                    },
                    "cost": 0.00042,
                    "cost_details": {
                        "upstream_inference_cost": 0.00031,
                    },
                    "is_byok": True,
                    "server_tool_use": {
                        "web_search_requests": 1,
                    },
                },
            },
        )

    provider, client = _build_provider(handler)

    try:
        generation = await provider.generate_async(
            model="openai/o3-mini",
            messages=[UserMessage(parts=[TextPart(text="hi")])],
            generation_config=GenerationConfig(
                reasoning={"effort": "high", "exclude": True}
            ),
        )
    finally:
        await client.aclose()

    assert captured_request_body["reasoning"] == {"effort": "high", "exclude": True}
    assert generation.message.text == "Hello from OpenRouter"
    assert generation.usage.prompt_tokens == 10
    assert generation.usage.prompt_tokens_details == {
        "cached_tokens": 4,
        "cache_write_tokens": 2,
    }
    assert generation.usage.completion_tokens_details == {
        "reasoning_tokens": 3,
        "image_tokens": 1,
    }
    assert generation.usage.cost == 0.00042
    assert generation.usage.cost_details == {
        "upstream_inference_cost": 0.00031,
    }
    assert generation.usage.is_byok is True
    assert generation.usage.server_tool_use == {"web_search_requests": 1}
    assert generation.usage.raw_usage["cost"] == 0.00042


def _observability_body(trace_params):
    request_body = {}
    OpenRouterGenerationProvider._apply_observability_params(
        request_body, GenerationConfig(trace_params=trace_params)
    )
    return request_body


def test_observability_params_send_environment_release_tags_and_metadata_in_trace():
    request_body = _observability_body(
        {
            "name": "Correcao",
            "user_id": "consumer",
            "session_id": "session",
            "environment": "production",
            "release": "1.2.3",
            "tags": ["redacao", "lote"],
            "metadata": {"step": "etapa", "passagem": 2},
        }
    )

    assert request_body["user"] == "consumer"
    assert request_body["session_id"] == "session"
    assert request_body["metadata"] == {"step": "etapa", "passagem": "2"}
    assert request_body["trace"] == {
        "trace_name": "Correcao",
        "environment": "production",
        "release": "1.2.3",
        "tags": "redacao,lote",
        "step": "etapa",
        "passagem": "2",
    }


def test_observability_metadata_does_not_override_reserved_trace_keys():
    request_body = _observability_body(
        {
            "trace_name": "Nome real",
            "tags": ["real"],
            "metadata": {"trace_name": "Outro nome", "tags": "outra"},
        }
    )

    assert request_body["trace"] == {"trace_name": "Nome real", "tags": "real"}


def test_observability_user_is_capped_at_broadcast_limit():
    request_body = _observability_body({"user_id": "u" * 200})

    assert request_body["user"] == "u" * 128


@pytest.mark.asyncio
async def test_stream_async_sends_reasoning_config_and_preserves_reasoning_details():
    captured_request_body = {}
    reasoning_details = [
        {
            "type": "reasoning.text",
            "text": "Let me think through this first.",
            "format": "anthropic-claude-v1",
            "id": "reasoning-text-1",
            "index": 0,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request_body
        captured_request_body = json.loads(request.content.decode("utf-8"))
        chunk = {
            "id": "resp_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/o3-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello",
                        "reasoning": "Let me think through this first.",
                        "reasoning_details": reasoning_details,
                    }
                }
            ],
        }
        usage_chunk = {
            "id": "resp_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/o3-mini",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 6,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.00025,
                "cost_details": {"upstream_inference_cost": 0.0002},
                "server_tool_use": {"web_search_requests": 2},
            },
        }
        content = (
            f"data: {json.dumps(chunk)}\n\n"
            f"data: {json.dumps(usage_chunk)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=content,
            headers={"Content-Type": "text/event-stream"},
        )

    provider, client = _build_provider(handler)

    try:
        chunks = [
            chunk
            async for chunk in provider.stream_async(
                model="openai/o3-mini",
                messages=[UserMessage(parts=[TextPart(text="hi")])],
                generation_config=GenerationConfig(reasoning={"max_tokens": 1200}),
            )
        ]
    finally:
        await client.aclose()

    assert captured_request_body["reasoning"] == {"max_tokens": 1200}
    assert chunks
    assert chunks[-1].message.text == "Hello"
    assert chunks[-1].message.reasoning == "Let me think through this first."
    assert chunks[-1].message.reasoning_details == reasoning_details
    assert chunks[-1].usage.prompt_tokens == 11
    assert chunks[-1].usage.prompt_tokens_details == {"cached_tokens": 5}
    assert chunks[-1].usage.completion_tokens_details == {"reasoning_tokens": 4}
    assert chunks[-1].usage.cost == 0.00025
    assert chunks[-1].usage.cost_details == {"upstream_inference_cost": 0.0002}
    assert chunks[-1].usage.server_tool_use == {"web_search_requests": 2}


@pytest.mark.asyncio
async def test_agent_roundtrip_preserves_reasoning_back_to_openrouter_on_tool_call():
    requests = []
    reasoning_details = [
        {
            "type": "reasoning.summary",
            "summary": "Need the weather tool before answering.",
            "format": "anthropic-claude-v1",
            "id": "reasoning-summary-1",
            "index": 0,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content.decode("utf-8"))
        requests.append(request_body)

        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "openai/o3-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "Need the weather tool before answering.",
                                "reasoning_details": reasoning_details,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location":"Boston"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 10,
                        "total_tokens": 25,
                    },
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "resp_2",
                "object": "chat.completion",
                "created": 2,
                "model": "openai/o3-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Bring a jacket, it is rainy in Boston.",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            },
        )

    def get_weather(location: str) -> str:
        return json.dumps({"location": location, "condition": "rainy"})

    provider, client = _build_provider(handler)
    agent = Agent(
        generation_provider=provider,
        model="openai/o3-mini",
        instructions="Use tools when needed.",
        tools=[get_weather],
        config=AgentConfig(
            generationConfig=GenerationConfig(reasoning={"effort": "medium"})
        ),
    )

    try:
        output = await agent.run_async("What should I wear in Boston today?")
    finally:
        await client.aclose()

    assert output.text == "Bring a jacket, it is rainy in Boston."
    assert len(requests) == 2
    assert requests[0]["reasoning"] == {"effort": "medium"}
    assert requests[1]["reasoning"] == {"effort": "medium"}

    assistant_messages = [
        message for message in requests[1]["messages"] if message["role"] == "assistant"
    ]
    assert assistant_messages
    assert assistant_messages[0]["reasoning"] == (
        "Need the weather tool before answering."
    )
    assert assistant_messages[0]["reasoning_details"] == reasoning_details
    assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
