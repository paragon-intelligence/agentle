from __future__ import annotations

from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.parsing.image import Image
from agentle.parsing.page_item.heading_page_item import HeadingPageItem
from agentle.parsing.page_item.table_page_item import TablePageItem
from agentle.parsing.page_item.text_page_item import TextPageItem


class SectionContent(BaseModel):
    """
    Represents a section of content within a parsed document.

    A SectionContent object represents a logical section or division within a document,
    such as a page in a PDF, a slide in a presentation, or a worksheet in a spreadsheet.
    Each section contains text content, optional markdown representation, images found
    within the section, and structured page items (like headings, text blocks, and tables).

    **Attributes:**

    *   `number` (int):
        Section number or index, used to identify and order sections within a document.

        **Example:**
        ```python
        section = SectionContent(number=1, text="Section content")
        print(section.number)  # Output: 1
        ```

    *   `text` (str):
        The raw text content of the section.

        **Example:**
        ```python
        section = SectionContent(number=1, text="This is the content of section 1.")
        print(section.text)  # Output: This is the content of section 1.
        ```

    *   `md` (str | None):
        Optional markdown representation of the section content. This can include
        formatted text, lists, code blocks, etc. Defaults to None.

        **Example:**
        ```python
        section = SectionContent(
            number=1,
            text="Section with header",
            md="# Section with header"
        )
        print(section.md)  # Output: # Section with header
        ```

    *   `images` (Sequence[Image]):
        A sequence of Image objects found within this section. Defaults to an empty list.

        **Example:**
        ```python
        from agentle.parsing.image import Image

        image = Image(name="figure1.png", contents=b"...", ocr_text="Image text")
        section = SectionContent(number=1, text="Section with image", images=[image])

        for img in section.images:
            print(f"Image: {img.name}, OCR: {img.ocr_text}")
        ```

    *   `items` (Sequence[TextPageItem | HeadingPageItem | TablePageItem]):
        A sequence of structured page items within this section. These represent
        specific content elements like headings, text blocks, and tables. Defaults to an empty list.

        **Example:**
        ```python
        from agentle.parsing.page_item.heading_page_item import HeadingPageItem
        from agentle.parsing.page_item.text_page_item import TextPageItem

        heading = HeadingPageItem(heading="Introduction", lvl=1)
        paragraph = TextPageItem(text="This is the introduction paragraph.")

        section = SectionContent(
            number=1,
            text="Introduction\nThis is the introduction paragraph.",
            items=[heading, paragraph]
        )
        ```

    **Usage Examples:**

    Creating a simple section:
    ```python
    section = SectionContent(
        number=1,
        text="Introduction\nThis is the first section of the document.",
        md="# Introduction\n\nThis is the first section of the document."
    )
    ```

    Creating a section with images and structured items:
    ```python
    from agentle.parsing.image import Image
    from agentle.parsing.page_item.heading_page_item import HeadingPageItem
    from agentle.parsing.page_item.text_page_item import TextPageItem
    from agentle.parsing.page_item.table_page_item import TablePageItem

    # Create images
    image1 = Image(name="chart.png", contents=b"...", ocr_text="Sales Chart 2023")

    # Create structured items
    heading = HeadingPageItem(heading="Sales Report", lvl=1)
    intro = TextPageItem(text="This report summarizes our 2023 sales figures.")
    table = TablePageItem(
        rows=[["Q1", "Q2", "Q3", "Q4"], ["$10K", "$12K", "$15K", "$18K"]],
        csv="Q1,Q2,Q3,Q4\n$10K,$12K,$15K,$18K"
    )

    # Create the section
    section = SectionContent(
        number=1,
        text="Sales Report\nThis report summarizes our 2023 sales figures.\n[Table with quarterly sales]",
        md="# Sales Report\n\nThis report summarizes our 2023 sales figures.\n\n[Table with quarterly sales]",
        images=[image1],
        items=[heading, intro, table]
    )
    ```
    """

    number: int = Field(
        description="Section number",
    )

    text: str = Field(
        description="Text content's of the page",
    )

    md: str | None = Field(
        default=None,
        description="Markdown representation of the section.",
    )

    images: Sequence[Image] = Field(
        default_factory=list,
        description="Images present in the section",
    )

    items: Sequence[TextPageItem | HeadingPageItem | TablePageItem] = Field(
        default_factory=list,
        description="Items present in the page",
    )

    def get_id(self) -> str:
        """
        Generate a unique identifier for this section.

        Returns:
            str: A unique identifier string in the format "page_{number}"

        Example:
            ```python
            section = SectionContent(number=3, text="Section content")
            section_id = section.get_id()
            print(section_id)  # Output: "page_3"
            ```
        """
        return f"page_{self.number}"

    def __add__(self, other: SectionContent) -> SectionContent:
        """
        Combine two section contents into a single section.

        This operator allows merging two sections by combining their text, markdown,
        images, and items. The section number of the resulting section will be the
        same as the left operand.

        Args:
            other (SectionContent): Another section to merge with this one

        Returns:
            SectionContent: A new section containing the combined content of both sections

        Example:
            ```python
            section1 = SectionContent(
                number=1,
                text="First part",
                images=[Image(name="img1.png", contents=b"...")]
            )

            section2 = SectionContent(
                number=2,
                text="Second part",
                images=[Image(name="img2.png", contents=b"...")]
            )

            combined = section1 + section2
            print(combined.number)  # Output: 1 (keeps the number from section1)
            print(combined.text)    # Output: First partSecond part
            print(len(combined.images))  # Output: 2
            ```
        """
        from itertools import chain

        return SectionContent(
            number=self.number,
            text=self.text + other.text,
            md=(self.md or "") + (other.md or ""),
            images=list(chain(self.images, other.images)),
            items=list(chain(self.items, other.items)),
        )
