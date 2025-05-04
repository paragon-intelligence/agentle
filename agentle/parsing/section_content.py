from __future__ import annotations

from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.parsing.image import Image
from agentle.parsing.page_item.heading_page_item import HeadingPageItem
from agentle.parsing.page_item.table_page_item import TablePageItem
from agentle.parsing.page_item.text_page_item import TextPageItem


class SectionContent(BaseModel):
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
        return f"page_{self.number}"

    def __add__(self, other: SectionContent) -> SectionContent:
        from itertools import chain

        return SectionContent(
            number=self.number,
            text=self.text + other.text,
            md=(self.md or "") + (other.md or ""),
            images=list(chain(self.images, other.images)),
            items=list(chain(self.items, other.items)),
        )
