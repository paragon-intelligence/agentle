"""
Module for graphical element description models.

This module provides the GraphicalElementDescription class which represents detailed descriptions
of individual visual or auditory elements within media, including their characteristics,
roles, and relationships with other elements.
"""

from typing import Optional, Sequence

from pydantic import BaseModel, Field


class GraphicalElementDescription(BaseModel):
    """A description of a visual or auditory element within a media file.

    Attributes:
        type: The general category of this visual element within the media
        details: A detailed description of the characteristics and properties of this element
        role: The purpose or function of this element within the context of the media
        relationships: How this element is related to other elements in the media

    Example:
        >>> GraphicalElementDescription(
        ...     type="Text string",
        ...     details="The number '3' in the top-left corner",
        ...     role="Represents the coefficient of x",
        ...     relationships=["Located above the main equation"]
        ... )
    """

    type: str = Field(
        title="Element Type",
        description="The general category of this visual element within the media. This could be a recognizable object, a symbol, a graphical component, a section of text, or any other distinct visual or temporal component. Be descriptive but not necessarily tied to real-world objects if the media is abstract or symbolic. Examples: 'Equation term', 'Geometric shape', 'Timeline marker', 'Audio waveform segment', 'Brushstroke', 'Data point'.",
        examples=[
            "Text string",
            "Geometric shape",
            "Timeline marker",
            "Component of a machine",
            "Area of color",
            "Video transition",
        ],
    )

    details: str = Field(
        title="Element Details",
        description="A detailed description of the characteristics and properties of this element. Focus on what is visually or audibly apparent. For text, provide the content. For shapes, describe form, color, and features. For abstract elements, describe visual properties like color, texture, and form, or temporal properties like duration and transitions. Be specific and descriptive. Examples: 'The text string 'y = mx + c' in bold font', 'A red circle with a thick black outline', 'A sudden fade to black', 'A high-pitched tone'.",
        examples=[
            "The number '3' in the top-left corner",
            "A thin, dashed black line",
            "A vibrant green triangular shape",
            "A slow zoom-in effect",
            "A burst of static",
        ],
    )

    role: Optional[str] = Field(
        default=None,
        title="Element Role/Function",
        description="The purpose or function of this element within the context of the media. How does it contribute to the overall meaning, structure, or flow? For example, in a formula, describe its mathematical role. In a diagram, its function. In a video, its narrative or informational contribution. Examples: 'Represents a variable in the equation', 'Indicates the direction of flow', 'Marks a key event in the timeline', 'Signals a change in scene'.",
        examples=[
            "Represents the coefficient of x",
            "Connects two stages in the process",
            "Highlights a critical moment",
            "Provides context for the following scene",
        ],
    )

    relationships: Optional[Sequence[str]] = Field(
        default=None,
        title="Element Relationships",
        description="Describe how this element is related to other elements in the media. "
        + "Explain its position relative to others, whether it's connected, overlapping, "
        + "near, or otherwise associated with them, considering spatial and temporal "
        + "relationships. Be specific about the other elements involved. Examples: "
        + "'The arrow points from this box to the next', 'This circle is enclosed"
        + "within the square', 'This scene follows the previous one', 'The music"
        + "swells during this visual element'.",
        examples=[
            "Located above the main equation",
            "Connected to the previous step by a line",
            "Part of a larger assembly",
            "Occurs immediately after the title card",
        ],
    )

    extracted_text: Optional[str] = Field(
        default=None,
        title="Extracted Text Content",
        description="For elements that contains text elements, the actual textual content "
        + "extracted through OCR. Preserves line breaks and spatial relationships where "
        + "possible.",
        examples=["'3.14'", "'Warning: Do not open'", "'y = mx + b'"],
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
