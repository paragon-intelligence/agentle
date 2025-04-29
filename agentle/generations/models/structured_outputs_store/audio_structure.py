"""
Module for audio structure models.

This module provides the AudioStructure class which represents the organizational structure
of audio content, including how elements are arranged, grouped, and emphasized.
"""

from typing import Optional, Sequence

from pydantic import BaseModel, Field


class AudioStructure(BaseModel):
    """A description of the overall structure and organization of an audio media file.

    Attributes:
        organization: A description of how the audio elements are arranged and organized within the media
        groupings: Significant groupings of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> AudioStructure(
        ...     organization="A narrative with a clear beginning, middle, and end",
        ...     groupings=["The introduction of the song", "The main argument of the speech"],
        ...     focal_point="The main theme of the music"
        ... )
    """

    organization: Optional[str] = Field(
        default=None,
        title="Overall Organization",
        description="A description of how the audio elements are arranged and organized within the media. Describe the overall structure, flow, or pattern. Is it linear, cyclical, thematic, or something else? How are the different parts connected or separated? Examples: 'A song with verse-chorus structure', 'A chronological sequence of spoken events', 'A layered soundscape with overlapping elements'.",
        examples=[
            "A narrative with a clear beginning, middle, and end",
            "A repeating musical motif",
            "A conversation with alternating speakers",
        ],
    )

    groupings: Optional[Sequence[str]] = Field(
        default=None,
        title="Significant Groupings of Elements",
        description="Describe any notable groupings or clusters of audio elements that appear to function together or have a shared context. Explain what binds these elements together aurally or conceptually. Examples: 'The instrumental section of the song', 'A dialogue between two characters', 'A series of related sound effects'.",
        examples=[
            "The introduction of the song",
            "The main argument of the speech",
            "The sounds of a busy street",
        ],
    )

    focal_point: Optional[str] = Field(
        default=None,
        title="Primary Focal Point",
        description="Identify the most prominent or central audio element or section that draws the listener's attention. Explain why this element stands out (e.g., volume, pitch, prominence of a voice or instrument). If there isn't a clear focal point, describe the distribution of auditory emphasis. Examples: 'The lead vocalist's melody', 'The loudest sound effect', 'The central argument of the speech'.",
        examples=[
            "The main theme of the music",
            "The key statement in the narration",
            "A sudden loud bang",
        ],
    )

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.organization:
            md_str += f"{indent}**Overall Organization**: {self.organization}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str
