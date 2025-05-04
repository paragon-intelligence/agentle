"""
Module for audio description models.

This module provides the AudioDescription class which represents detailed descriptions
of audio content, including its overall characteristics and features.
"""

from collections.abc import Sequence
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.structured_outputs_store.audio_element_description import (
    AudioElementDescription,
)
from agentle.generations.models.structured_outputs_store.audio_structure import (
    AudioStructure,
)


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

    content_type: str = Field(
        description="A general categorization of the audio's content. This helps to broadly define what kind of auditory experience or information is being presented. Examples: 'Podcast', 'Song', 'Speech', 'Sound effects', 'Ambient music', 'Audiobook', 'Interview'.",
        examples=["Podcast", "Music", "Speech", "Sound Effects"],
    )

    audio_elements: Sequence[AudioElementDescription] | None = Field(
        default=None,
        description="A list of individual audio elements identified within the media, each with its own detailed description. For each element, provide its type, specific auditory details, its role or function within the audio's context, and its relationships to other elements. The goal is to break down the audio into its fundamental auditory components and describe them comprehensively. This applies to all types of audio, from spoken words in a podcast to musical notes in a song or distinct sound effects.",
    )

    structure: AudioStructure | None = Field(
        default=None,
        description="A description of the overall structure and organization of the audio elements within the media. This section focuses on how the different parts are arranged and related to each other. Describe the overall organization, any significant groupings of elements, and the primary focal point or area of emphasis. This helps to understand the higher-level organization of the audio's content.",
    )

    dominant_auditory_features: Sequence[str] | None = Field(
        default=None,
        description="A list of the most striking auditory features of the audio that contribute significantly to its overall character and impact. This could include dominant melodies, rhythmic patterns, distinctive voices or timbres, recurring sound effects, or any other salient auditory characteristics. Be specific and descriptive. Examples: 'A strong, repetitive beat', 'A high-pitched, clear female voice', 'Frequent use of echo and reverb', 'A melancholic piano melody'.",
        examples=[
            "A fast tempo",
            "A deep bassline",
            "Clear and articulate speech",
        ],
    )

    intended_purpose: str | None = Field(
        default=None,
        description="An interpretation of the intended purpose or meaning of the audio, based on its content and structure. What is the audio trying to convey or communicate? For a song, it might be to express emotions. For a podcast, to inform or entertain. For sound effects, to create a specific atmosphere. This is an interpretive field, so focus on reasonable inferences based on the auditory evidence. Examples: 'To tell a story through sound', 'To provide information on a specific topic', 'To create a relaxing and immersive soundscape', 'To evoke feelings of joy and excitement'.",
        examples=[
            "To entertain the listener",
            "To educate on a particular subject",
            "To create a sense of atmosphere",
        ],
    )

    @property
    def md(self) -> str:
        md_str = ""
        md_str += f"## Overall Audio Description\n{self.overall_description}\n\n"
        md_str += f"## Content Type\n{self.content_type}\n\n"

        if self.audio_elements:
            md_str += "## Detailed Audio Element Descriptions\n"
            for element in self.audio_elements:
                md_str += element.md(indent_level=0) + "\n"

        if self.structure:
            md_str += "## Audio Structure and Organization\n"
            md_str += self.structure.md(indent_level=1) + "\n"

        if self.dominant_auditory_features:
            md_str += "## Dominant Auditory Features\n"
            for feature in self.dominant_auditory_features:
                md_str += f"- {feature}\n"
            md_str += "\n"

        if self.intended_purpose:
            md_str += f"## Intended Purpose or Meaning\n{self.intended_purpose}\n\n"

        return md_str
