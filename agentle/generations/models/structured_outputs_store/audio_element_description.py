"""
Module for audio element description models.

This module provides the AudioElementDescription class which represents detailed descriptions
of individual components within audio content, including their characteristics and relationships.
"""

from typing import Optional, Sequence

from pydantic import BaseModel, Field


class AudioElementDescription(BaseModel):
    """A description of an audio element within a media file.

    Attributes:
        type: The general category of this audio element within the media
        details: A detailed description of the auditory characteristics and properties of this element
        role: The purpose or function of this audio element within the context of the media
        relationships: How this audio element is related to other elements in the media

    Example:
        >>> AudioElementDescription(
        ...     type="Speech segment",
        ...     details="The word 'example' spoken with emphasis",
        ...     role="Introduces the main subject",
        ...     relationships=["Occurs after a period of silence"]
        ... )
    """

    type: str = Field(
        title="Element Type",
        description="The general category of this audio element within the media. This could be a type of sound, a segment of speech, a musical phrase, or any other distinct auditory component. Examples: 'Speech segment', 'Musical note', 'Sound effect', 'Silence', 'Jingle'.",
        examples=[
            "Speech",
            "Melody",
            "Footsteps",
            "Silence",
        ],
    )

    details: str = Field(
        title="Element Details",
        description="A detailed description of the auditory characteristics and properties of this element. For speech, provide the content or a description of the speaker's tone and delivery. For music, describe the melody, harmony, rhythm, and instrumentation. For sound effects, describe the sound and its characteristics. Be specific and descriptive. Examples: 'The spoken phrase 'Hello world' in a clear voice', 'A high-pitched sustained note on a violin', 'The sound of a door slamming shut', 'A brief period of complete silence'.",
        examples=[
            "The word 'example' spoken with emphasis",
            "A low humming sound",
            "A sharp, percussive beat",
        ],
    )

    role: Optional[str] = Field(
        default=None,
        title="Element Role/Function",
        description="The purpose or function of this audio element within the context of the media. How does it contribute to the overall meaning, mood, or structure? For example, in a song, describe its role in the melody or harmony. In a spoken piece, explain its informational or emotional contribution. In a soundscape, its contribution to the atmosphere. Examples: 'Conveys information about the topic', 'Creates a sense of tension', 'Marks the beginning of a new section', 'Provides background ambience'.",
        examples=[
            "Introduces the main subject",
            "Builds suspense",
            "Signals a transition",
            "Establishes the setting",
        ],
    )

    relationships: Optional[Sequence[str]] = Field(
        default=None,
        title="Element Relationships",
        description="Describe how this audio element is related to other elements in the media. Explain its temporal relationship to others, whether it occurs before, during, or after other sounds, or how it interacts with other auditory elements. Be specific about the other elements involved. Examples: 'This musical phrase follows the introductory melody', 'The sound effect occurs simultaneously with the visual impact', 'The speaker's voice overlaps with the background music'.",
        examples=[
            "Occurs after a period of silence",
            "Plays under the main narration",
            "A response to the previous sound",
        ],
    )

    def md(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        md_str = f"{indent}**Element Type**: {self.type}\n"
        md_str += f"{indent}**Element Details**: {self.details}\n"
        if self.role:
            md_str += f"{indent}**Role/Function**: {self.role}\n"
        if self.relationships:
            md_str += f"{indent}**Relationships**:\n"
            for rel in self.relationships:
                md_str += f"{indent}  - {rel}\n"
        return md_str
