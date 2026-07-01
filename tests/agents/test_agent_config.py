from agentle.agents.agent_config import AgentConfig


def test_agent_config_passes_google_generation_config_dict():
    config = AgentConfig(
        generationConfig={
            "seed": 123,
            "google": {
                "cached_content": "cachedContents/agent",
                "response_modalities": ["TEXT"],
            },
        }
    )

    generation_config = config.generation_config

    assert generation_config.seed == 123
    assert generation_config.google is not None
    assert generation_config.google.cached_content == "cachedContents/agent"
    assert generation_config.google.response_modalities == ["TEXT"]
