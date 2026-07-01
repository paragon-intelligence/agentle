import pytest

from agentle.generations.models.generation.generation_config import GenerationConfig


def test_generation_config_accepts_reasoning_dict():
    config = GenerationConfig(reasoning={"effort": "high", "exclude": True})

    assert config.reasoning is not None
    assert config.reasoning.effort == "high"
    assert config.reasoning.exclude is True


def test_generation_config_rejects_effort_and_max_tokens_together():
    with pytest.raises(
        ValueError,
        match="Only one of reasoning.effort or reasoning.max_tokens should be set.",
    ):
        GenerationConfig(reasoning={"effort": "high", "max_tokens": 2048})


def test_generation_config_clone_preserves_reasoning():
    config = GenerationConfig(
        reasoning={"effort": "medium"},
        trace_params={"name": "reasoning-test"},
    )

    cloned = config.clone(new_trace_params={"session_id": "session-1"})

    assert cloned.reasoning is not None
    assert cloned.reasoning.effort == "medium"
    assert cloned.trace_params["name"] == "reasoning-test"
    assert cloned.trace_params["session_id"] == "session-1"


def test_generation_config_accepts_common_google_config_dict():
    config = GenerationConfig(
        stop_sequences=["END"],
        seed=123,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        logprobs=3,
        response_logprobs=True,
        google={
            "cached_content": "cachedContents/test",
            "response_modalities": ["TEXT"],
            "service_tier": "standard",
        },
    )

    assert config.stop_sequences == ["END"]
    assert config.seed == 123
    assert config.presence_penalty == 0.1
    assert config.frequency_penalty == 0.2
    assert config.logprobs == 3
    assert config.response_logprobs is True
    assert config.google is not None
    assert config.google.cached_content == "cachedContents/test"
    assert config.google.response_modalities == ["TEXT"]
    assert config.google.service_tier == "standard"


def test_generation_config_clone_preserves_common_and_google_fields():
    config = GenerationConfig(
        stop_sequences=["END"],
        seed=123,
        google={"cached_content": "cachedContents/test"},
    )

    cloned = config.clone(new_seed=456)

    assert cloned.seed == 456
    assert cloned.stop_sequences == ["END"]
    assert cloned.google is not None
    assert cloned.google.cached_content == "cachedContents/test"
