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

    **Inheritance:**

    *   Inherits from `PageItem`.

    **Attributes:**

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

    *   `md` (str):
        Inherited from `PageItem`. Markdown representation of the heading.
    """
    type: Literal["heading"] = Field(default="heading")

    heading: str = Field(
        description="Value of the heading",
    )

    lvl: int = Field(
        description="Level of the heading",
    )

    model_config = ConfigDict(frozen=True)
