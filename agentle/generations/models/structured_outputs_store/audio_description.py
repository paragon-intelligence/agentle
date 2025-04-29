"""
Module for audio description models.

This module provides the AudioDescription class which represents detailed descriptions
of audio content, including its overall characteristics and features.
"""

from pydantic import BaseModel, Field


class AudioDescription(BaseModel):
    """Detailed description of audio content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (podcast, music, etc.)
        audio_elements: Individual components
        structure: Spatial organization
        dominant_auditory_features: Salient auditory characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = AudioDescription(
        ...     content_type="podcast",
        ...     audio_elements=[...]
        ... )
    """

    overall_description: str = Field(
        title="Overall Audio Description",
        description="Provide a comprehensive and detailed narrative describing the entire audio media, focusing on its content, structure, and key auditory elements. Imagine you are explaining the audio to someone who cannot hear it. Describe the overall purpose or what information the audio is conveying or what experience it aims to create. Detail the main components and how they are organized. Use precise language to describe auditory characteristics like pitch, tone, rhythm, tempo, and instrumentation. For abstract audio, focus on describing the sonic properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure of the audio. Examples: 'The audio presents a news report detailing recent events, featuring a clear and professional narration with background music.', 'The audio is a piece of ambient music featuring layered synthesizers and natural soundscapes, creating a calming atmosphere.', 'The audio recording captures a lively conversation between two individuals, with distinct voices and occasional laughter.'",
        examples=[
            "A podcast discussing current events",
            "A musical piece with a strong melody",
            "A recording of nature sounds",
        ],
    )
