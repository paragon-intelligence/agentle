"""
Factory Functions for Creating Document Processing Agents

This package contains factory functions that create specialized agent instances
for different document processing tasks. These factories provide a convenient way
to instantiate properly configured agents with appropriate models and providers.

Available factories:
- audio_description_agent_default_factory: Creates an agent for audio file description and transcription
- visual_description_agent_default_factory: Creates an agent for analyzing and describing visual media

These factory functions abstract away the complexity of setting up the correct
models, instructions, and generation providers for specific document processing tasks.

Example usage:
```python
from agentle.parsing.factories import audio_description_agent_default_factory

# Create an audio description agent
audio_agent = audio_description_agent_default_factory()

# Use the agent to process an audio file
result = audio_agent.run(audio_file)
```
"""
