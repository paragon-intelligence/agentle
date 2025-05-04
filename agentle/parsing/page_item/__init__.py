"""
Page Item Types for Parsed Document Content

This package contains classes representing different types of content items
that can be found within a parsed document section. These items provide a structured
way to represent and work with specific content types like:

- Text blocks (TextPageItem)
- Headings/titles (HeadingPageItem)
- Tables (TablePageItem)

Each page item type is designed to capture the specific properties and characteristics
of that content type, making it easier to process and interact with document content
in a structured way.

Example usage:
```python
from agentle.parsing.page_item.heading_page_item import HeadingPageItem
from agentle.parsing.page_item.text_page_item import TextPageItem
from agentle.parsing.page_item.table_page_item import TablePageItem

# Create different page items
heading = HeadingPageItem(heading="Introduction", lvl=1)
text = TextPageItem(text="This is a paragraph of content.")
table = TablePageItem(
    rows=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
    csv="Header 1,Header 2\nData 1,Data 2"
)

# These items can be added to a SectionContent object
from agentle.parsing.section_content import SectionContent

section = SectionContent(
    number=1,
    text="Full section text...",
    items=[heading, text, table]
)
```
"""
