"""
Module for media structure models.

This module provides the MediaStructure class which represents the organizational structure
of visual or auditory media, including layout, groupings of elements, and focal points.
"""

from typing import Optional, Sequence

from pydantic import BaseModel, Field

from agentle.generations.models.structured_outputs_store.audio_element_description import (
    AudioElementDescription,
)
from agentle.generations.models.structured_outputs_store.audio_structure import (
    AudioStructure,
)
from agentle.generations.models.structured_outputs_store.graphical_element_description import (
    GraphicalElementDescription,
)


class VisualMediaDescription(BaseModel):
    """Detailed description of visual content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (diagram, photo, etc.)
        visual_elements: Individual components
        structure: Spatial organization
        dominant_features: Salient visual characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = VisualMediaDescription(
        ...     content_type="infographic",
        ...     visual_elements=[...]
        ... )
    """

    overall_description: str = Field(
        title="Overall Media Description",
        description="Provide a comprehensive and detailed narrative describing the entire visual media, focusing on its content, structure, and key elements. Imagine you are explaining it to someone who cannot see or hear it. Describe the overall purpose or what information it is conveying. Detail the main components and how they are organized, considering both spatial and temporal aspects. Use precise language to describe visual characteristics like shapes, colors, patterns, and relationships, as well as temporal characteristics like duration, transitions, and pacing. For abstract media, focus on describing the properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure. Examples: 'The video presents a step-by-step tutorial on assembling a device. Text overlays accompany the visual demonstrations.', 'The animated graphic shows the flow of data through a network, with arrows indicating direction and color-coding representing different types of data.', 'The abstract animation features pulsating colors and evolving geometric shapes set to a rhythmic soundtrack.'",
        examples=[
            "A diagram illustrating the water cycle",
            "A complex algebraic equation",
            "An abstract painting with bold colors",
            "A short film depicting a historical event",
        ],
    )

    content_type: str = Field(
        title="Content Type",
        description="A general categorization of the audio's content. This helps to broadly define what kind of auditory experience or information is being presented. Examples: 'Podcast', 'Song', 'Speech', 'Sound effects', 'Ambient music', 'Audiobook', 'Interview'.",
        examples=["Podcast", "Music", "Speech", "Sound Effects"],
    )

    audio_elements: Optional[Sequence[AudioElementDescription]] = Field(
        default=None,
        title="Detailed Audio Element Descriptions",
        description="A list of individual audio elements identified within the media, each with its own detailed description. For each element, provide its type, specific auditory details, its role or function within the audio's context, and its relationships to other elements. The goal is to break down the audio into its fundamental auditory components and describe them comprehensively. This applies to all types of audio, from spoken words in a podcast to musical notes in a song or distinct sound effects.",
    )

    visual_elements: Optional[Sequence[GraphicalElementDescription]] = Field(
        default=None, description="Visual elements"
    )

    dominant_features: Optional[Sequence[str]] = Field(
        default=None, description="Dominant features of the Visual media"
    )

    structure: Optional[AudioStructure] = Field(
        default=None,
        title="Audio Structure and Organization",
        description="A description of the overall structure and organization of the audio elements within the media. This section focuses on how the different parts are arranged and related to each other. Describe the overall organization, any significant groupings of elements, and the primary focal point or area of emphasis. This helps to understand the higher-level organization of the audio's content.",
    )

    dominant_auditory_features: Optional[Sequence[str]] = Field(
        default=None,
        title="Dominant Auditory Features",
        description="A list of the most striking auditory features of the audio that contribute significantly to its overall character and impact. This could include dominant melodies, rhythmic patterns, distinctive voices or timbres, recurring sound effects, or any other salient auditory characteristics. Be specific and descriptive. Examples: 'A strong, repetitive beat', 'A high-pitched, clear female voice', 'Frequent use of echo and reverb', 'A melancholic piano melody'.",
        examples=[
            "A fast tempo",
            "A deep bassline",
            "Clear and articulate speech",
        ],
    )

    intended_purpose: str | None = Field(
        default=None,
        title="Intended Purpose or Meaning",
        description="An interpretation of the intended purpose or meaning of the audio, based on its content and structure. What is the audio trying to convey or communicate? For a song, it might be to express emotions. For a podcast, to inform or entertain. For sound effects, to create a specific atmosphere. This is an interpretive field, so focus on reasonable inferences based on the auditory evidence. Examples: 'To tell a story through sound', 'To provide information on a specific topic', 'To create a relaxing and immersive soundscape', 'To evoke feelings of joy and excitement'.",
        examples=[
            "To entertain the listener",
            "To educate on a particular subject",
            "To create a sense of atmosphere",
        ],
    )

    ocr_text: str | None = Field(
        default=None,
        title="OCR Text",
        description="The OCR text of the visual media, if any.",
    )

    @property
    def md(self) -> str:
        md_str = f"## Overall Media Description\n{self.overall_description}\n\n"
        md_str += f"## Content Type\n{self.content_type}\n\n"

        if self.visual_elements:
            md_str += "## Detailed Element Descriptions\n"
            for element in self.visual_elements:
                md_str += element.md(indent_level=0) + "\n"

        if self.structure:
            md_str += "## Media Structure and Organization\n"
            md_str += self.structure.md(indent_level=1) + "\n"

        if self.dominant_features:
            md_str += "## Dominant Features\n"
            for feature in self.dominant_features:
                md_str += f"- {feature}\n"
            md_str += "\n"

        if self.intended_purpose:
            md_str += f"## Intended Purpose or Meaning\n{self.intended_purpose}\n\n"

        return md_str
