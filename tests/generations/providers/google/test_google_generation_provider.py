from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)


class _StructuredOutput(BaseModel):
    value: str


def _messages() -> list[DeveloperMessage | UserMessage]:
    return [
        DeveloperMessage(parts=[TextPart(text="You are concise.")]),
        UserMessage(parts=[TextPart(text="Say ok.")]),
    ]


def _response(types: Any) -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="ok")],
                )
            )
        ]
    )


def _install_fake_google_client(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Any]:
    from google import genai
    from google.genai import types

    captured: dict[str, Any] = {}

    class FakeModels:
        async def generate_content(
            self,
            *,
            model: str,
            contents: Any,
            config: Any,
        ) -> Any:
            captured["generate"] = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            return _response(types)

        async def generate_content_stream(
            self,
            *,
            model: str,
            contents: Any,
            config: Any,
        ) -> AsyncIterator[Any]:
            captured["stream"] = {
                "model": model,
                "contents": contents,
                "config": config,
            }

            async def stream() -> AsyncIterator[Any]:
                yield _response(types)

            return stream()

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(genai, "Client", FakeClient)
    return captured, types


def _config_dump(config: Any) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


async def test_generate_forwards_common_google_thinking_and_afc_config(
    monkeypatch: pytest.MonkeyPatch,
):
    captured, _ = _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(api_key="test-key")

    await provider.generate_async(
        messages=_messages(),
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=64,
            top_p=0.9,
            top_k=20,
            stop_sequences=["END"],
            seed=123,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            logprobs=3,
            response_logprobs=True,
            reasoning={"effort": "xhigh", "exclude": False},
            google={
                "automatic_function_calling": {
                    "disable": False,
                    "maximum_remote_calls": 4,
                    "ignore_call_history": True,
                },
                "cached_content": "cachedContents/test",
                "audio_timestamp": True,
                "media_resolution": "MEDIA_RESOLUTION_LOW",
                "response_modalities": ["TEXT"],
                "image_config": {"aspect_ratio": "1:1"},
                "labels": {"team": "agentle"},
                "service_tier": "standard",
            },
        ),
    )

    dump = _config_dump(captured["generate"]["config"])

    assert dump["temperature"] == 0.2
    assert dump["max_output_tokens"] == 64
    assert dump["top_p"] == 0.9
    assert dump["top_k"] == 20
    assert dump["stop_sequences"] == ["END"]
    assert dump["seed"] == 123
    assert dump["presence_penalty"] == 0.1
    assert dump["frequency_penalty"] == 0.2
    assert dump["logprobs"] == 3
    assert dump["response_logprobs"] is True
    assert dump["thinking_config"] == {
        "include_thoughts": True,
        "thinking_level": "HIGH",
    }
    assert dump["automatic_function_calling"] == {
        "disable": False,
        "maximum_remote_calls": 4,
        "ignore_call_history": True,
    }
    assert dump["cached_content"] == "cachedContents/test"
    assert dump["audio_timestamp"] is True
    assert dump["media_resolution"] == "MEDIA_RESOLUTION_LOW"
    assert dump["response_modalities"] == ["TEXT"]
    assert dump["image_config"]["aspect_ratio"] == "1:1"
    assert dump["labels"] == {"team": "agentle"}
    assert dump["service_tier"] == "standard"


async def test_stream_forwards_default_afc_and_reasoning_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    captured, _ = _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(
        api_key="test-key",
        function_calling_config={"disable": False, "ignore_call_history": True},
    )

    generations = [
        generation
        async for generation in provider.stream_async(
            messages=_messages(),
            generation_config=GenerationConfig(reasoning={"max_tokens": 1200}),
        )
    ]

    dump = _config_dump(captured["stream"]["config"])

    assert generations
    assert dump["thinking_config"] == {"thinking_budget": 1200}
    assert dump["automatic_function_calling"] == {
        "disable": False,
        "maximum_remote_calls": 10,
        "ignore_call_history": True,
    }


async def test_google_thinking_config_overrides_generic_reasoning(
    monkeypatch: pytest.MonkeyPatch,
):
    captured, _ = _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(api_key="test-key")

    await provider.generate_async(
        messages=_messages(),
        generation_config=GenerationConfig(
            reasoning={"effort": "high"},
            google={"thinking_config": {"thinking_budget": 0}},
        ),
    )

    dump = _config_dump(captured["generate"]["config"])

    assert dump["thinking_config"] == {"thinking_budget": 0}


async def test_google_response_json_schema_sets_json_mime_type(
    monkeypatch: pytest.MonkeyPatch,
):
    captured, _ = _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(api_key="test-key")

    await provider.generate_async(
        messages=_messages(),
        generation_config=GenerationConfig(
            google={
                "response_json_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
            }
        ),
    )

    dump = _config_dump(captured["generate"]["config"])

    assert dump["response_mime_type"] == "application/json"
    assert dump["response_json_schema"] == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }


async def test_response_schema_conflicts_with_google_response_json_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(api_key="test-key")

    with pytest.raises(
        ValueError,
        match="response_schema and generation_config.google.response_json_schema cannot both be set",
    ):
        await provider.generate_async(
            messages=_messages(),
            response_schema=_StructuredOutput,
            generation_config=GenerationConfig(
                google={"response_json_schema": {"type": "object"}}
            ),
        )


async def test_model_armor_config_conflicts_with_safety_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_google_client(monkeypatch)
    provider = GoogleGenerationProvider(api_key="test-key")

    with pytest.raises(
        ValueError,
        match="model_armor_config and generation_config.google.safety_settings cannot both be set",
    ):
        await provider.generate_async(
            messages=_messages(),
            generation_config=GenerationConfig(
                google={
                    "model_armor_config": {},
                    "safety_settings": [{}],
                }
            ),
        )


def test_vertex_ai_uses_enterprise_client_keyword(monkeypatch: pytest.MonkeyPatch):
    from google import genai

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(
            self,
            *,
            enterprise: bool | None = None,
            vertexai: bool | None = None,
            **kwargs: Any,
        ) -> None:
            captured["enterprise"] = enterprise
            captured["vertexai"] = vertexai
            captured["kwargs"] = kwargs
            self.aio = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr(genai, "Client", FakeClient)

    GoogleGenerationProvider(use_vertex_ai=True, project="project", location="global")

    assert captured["enterprise"] is True
    assert captured["vertexai"] is None
    assert captured["kwargs"]["project"] == "project"
    assert captured["kwargs"]["location"] == "global"
