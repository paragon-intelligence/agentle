"""
Heading Page Item Module

This module defines the HeadingPageItem class, which represents a heading within a
document section. It includes attributes for the heading text, level, and type.

"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from typing import Literal
from rsb.models.config_dict import ConfigDict


class HeadingPageItem(BaseModel):
    """
    Represents a heading within a document section.

    This class is a concrete implementation of `PageItem` and is used to represent
    headings (like section titles or subtitles) within a parsed document. It includes
    the heading text and the heading level (e.g., H1, H2, H3).

    **Attributes:**

    *   `type` (Literal["heading"]):
        Constant field that identifies this as a heading item. Always set to "heading".

        **Example:**
        ```python
        heading_item = HeadingPageItem(heading="Introduction", lvl=1)
        print(heading_item.type)  # Output: heading
        ```

    *   `heading` (str):
        The text content of the heading.

        **Example:**
        ```python
        heading_item = HeadingPageItem(md="## Section Title", heading="Section Title", lvl=2)
        print(heading_item.md)      # Output: ## Section Title
        print(heading_item.heading) # Output: Section Title
        print(heading_item.lvl)     # Output: 2
        print(heading_item.type)    # Output: heading
        ```

    *   `lvl` (int):
        The heading level (e.g., 1 for H1, 2 for H2, etc.).

        **Example:**
        ```python
        # Create headings of different levels
        h1 = HeadingPageItem(heading="Document Title", lvl=1)
        h2 = HeadingPageItem(heading="Chapter Title", lvl=2)
        h3 = HeadingPageItem(heading="Section Title", lvl=3)

        # Use the level to format differently
        for heading in [h1, h2, h3]:
            prefix = "#" * heading.lvl
            print(f"{prefix} {heading.heading}")
        ```

    **Usage Examples:**

    Creating headings of different levels:
    ```python
    from agentle.parsing.page_item.heading_page_item import HeadingPageItem

    # Create a primary heading (H1)
    title = HeadingPageItem(heading="Document Title", lvl=1)

    # Create a secondary heading (H2)
    chapter = HeadingPageItem(heading="Introduction", lvl=2)

    # Create a tertiary heading (H3)
    section = HeadingPageItem(heading="Background", lvl=3)
    ```

    Adding headings to a section:
    ```python
    from agentle.parsing.section_content import SectionContent
    from agentle.parsing.page_item.heading_page_item import HeadingPageItem
    from agentle.parsing.page_item.text_page_item import TextPageItem

    # Create headings and text content
    title = HeadingPageItem(heading="Report Title", lvl=1)
    intro_heading = HeadingPageItem(heading="Introduction", lvl=2)
    intro_text = TextPageItem(text="This report covers the findings of our research...")

    # Create a section with the headings and text
    section = SectionContent(
        number=1,
        text="Report Title\n\nIntroduction\n\nThis report covers the findings of our research...",
        items=[title, intro_heading, intro_text]
    )
    ```
    """

    type: Literal["heading"] = Field(default="heading")

    heading: str = Field(
        description="Value of the heading",
    )

    lvl: int = Field(
        description="Level of the heading",
    )

    model_config = ConfigDict(frozen=True)
