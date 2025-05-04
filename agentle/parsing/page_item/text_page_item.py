from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from typing import Literal
from rsb.models.config_dict import ConfigDict


class TextPageItem(BaseModel):
    """
    Represents a block of text within a document section.

    This class is a concrete implementation of a page item for representing
    plain text content within a parsed document. It is used for paragraphs,
    prose, or any other textual content that isn't a heading or structured data.

    **Attributes:**

    *   `type` (Literal["text"]):
        Constant field that identifies this as a text item. Always set to "text".

        **Example:**
        ```python
        text_item = TextPageItem(text="Sample text")
        print(text_item.type) # Output: text
        ```

    *   `text` (str):
        The text content of this item.

        **Example:**
        ```python
        text_item = TextPageItem(text="This is a paragraph of text.")
        print(text_item.text) # Output: This is a paragraph of text.
        ```

    **Usage Example:**

    ```python
    # Creating a basic text item
    paragraph = TextPageItem(text="This is a paragraph from the document.")

    # Adding to a section with other content types
    from agentle.parsing.section_content import SectionContent
    from agentle.parsing.page_item.heading_page_item import HeadingPageItem

    section = SectionContent(
        number=1,
        text="Section 1\nThis is a paragraph from the document.",
        items=[
            HeadingPageItem(heading="Section 1", lvl=1),
            paragraph
        ]
    )
    ```
    """

    type: Literal["text"] = Field(default="text")

    text: str = Field(
        description="Value of the text item",
    )

    model_config = ConfigDict(frozen=True)
