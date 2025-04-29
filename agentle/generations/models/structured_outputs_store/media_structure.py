"""
Module for visual media description models.

This module provides the VisualMediaDescription class which represents detailed descriptions
of visual content, including overall descriptions, elements, structure, and features.
"""

from typing import Optional, Sequence

from pydantic import BaseModel, Field


class MediaStructure(BaseModel):
    """A description of the overall structure and organization of a visual or auditory media file.

    Attributes:
        layout: A description of how the elements are arranged and organized within the media
        groupings: Significant groupings or clusters of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> MediaStructure(
        ...     layout="A step-by-step diagram",
        ...     groupings=["The main body of the text", "The elements forming the control panel"],
        ...     focal_point="The large heading at the top"
        ... )
    """

    layout: Optional[str] = Field(
        default=None,
        title="Overall Layout and Organization",
        description="A description of how the elements are arranged and organized within the media. Describe the overall structure, flow, or pattern, considering both spatial and temporal aspects. Is it linear, grid-based, hierarchical, sequential, or something else? How are the different parts connected or separated? Examples: 'A top-down flowchart', 'A grid of data points', 'A chronological sequence of scenes', 'A central diagram with surrounding labels'.",
        examples=[
            "A step-by-step diagram",
            "A clustered arrangement of shapes",
            "A formula presented on a single line",
            "A narrative with distinct acts",
        ],
    )

    groupings: Optional[Sequence[str]] = Field(
        default=None,
        title="Significant Groupings of Elements",
        description="Describe any notable groupings or clusters of elements that appear to function together or have a shared context, considering both visual and temporal coherence. Explain what binds these elements together visually, aurally, or conceptually. Examples: 'The terms on the left side of the equation', 'The interconnected components of the circuit diagram', 'A montage of related images', 'A musical theme associated with a character'.",
        examples=[
            "The main body of the text",
            "The elements forming the control panel",
            "The interconnected nodes of the network",
            "A series of shots depicting the same event",
        ],
    )

    focal_point: Optional[str] = Field(
        default=None,
        title="Primary Focal Point",
        description="Identify the most prominent or central element or area that draws attention, considering visual, auditory, and temporal emphasis. Explain why this element stands out (e.g., size, color, position, duration, sound intensity). If there isn't a clear focal point, describe the distribution of emphasis. Examples: 'The main title of the document', 'The central component of the machine', 'The climax of the scene', 'The loudest sound'.",
        examples=[
            "The large heading at the top",
            "The brightly colored area in the center",
            "The main subject of the drawing",
            "The key moment of impact",
        ],
    )

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.layout:
            md_str += f"{indent}**Overall Layout and Organization**: {self.layout}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str
