"""
Table Page Item Module

This module defines the TablePageItem class, which represents a table extracted
from a document section. It includes attributes for the table data, CSV representation,
and a flag indicating if the table is considered a "perfect" table.
"""

from collections.abc import Sequence
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class TablePageItem(BaseModel):
    """
    Represents a table extracted from a document section.

    This class is a concrete implementation of `PageItem` and is used to represent
    tables found within a parsed document. It includes the table data as a list of rows,
    a CSV representation of the table, and a flag indicating if the table is considered a "perfect" table
    (e.g., well-structured without irregularities).

    **Attributes:**

    *   `type` (Literal["table"]):
        Constant field that identifies this as a table item. Always set to "table".

        **Example:**
        ```python
        table_item = TablePageItem(rows=[["Header"]], csv="Header")
        print(table_item.type)  # Output: table
        ```

    *   `rows` (Sequence[Sequence[str]]):
        A sequence of rows, where each row is a sequence of strings representing the cells in that row.

        **Example:**
        ```python
        table_rows = [
            ["Header 1", "Header 2"],
            ["Data 1", "Data 2"],
            ["Data 3", "Data 4"]
        ]
        table_item = TablePageItem(md="| Header 1 | Header 2 |\n|---|---|\n| Data 1 | Data 2 |\n| Data 3 | Data 4 |", rows=table_rows, csv="Header 1,Header 2\\nData 1,Data 2\\nData 3,Data 4", is_perfect_table=True)
        print(table_item.rows) # Output: [['Header 1', 'Header 2'], ['Data 1', 'Data 2'], ['Data 3', 'Data 4']]
        ```

    *   `csv` (str):
        A string containing the CSV (Comma Separated Values) representation of the table data.

        **Example:**
        ```python
        print(table_item.csv) # Output: Header 1,Header 2\nData 1,Data 2\nData 3,Data 4
        ```

    *   `is_perfect_table` (bool):
        A boolean flag indicating whether the table is considered a "perfect table".
        This can be used to differentiate between well-structured tables and tables with potential irregularities.
        Defaults to `False`.

        **Example:**
        ```python
        print(table_item.is_perfect_table) # Output: True
        ```

    **Usage Examples:**

    Creating a simple table:
    ```python
    from agentle.parsing.page_item.table_page_item import TablePageItem

    # Create a basic table with headers and data
    table = TablePageItem(
        rows=[
            ["Name", "Age", "Occupation"],
            ["Alice", "28", "Engineer"],
            ["Bob", "34", "Designer"],
            ["Carol", "42", "Manager"]
        ],
        csv="Name,Age,Occupation\nAlice,28,Engineer\nBob,34,Designer\nCarol,42,Manager",
        is_perfect_table=True
    )

    # Access table data
    print(f"Table has {len(table.rows)} rows")
    print(f"Headers: {', '.join(table.rows[0])}")

    # Process table data
    for row in table.rows[1:]:  # Skip header row
        name, age, occupation = row
        print(f"{name} is a {occupation} who is {age} years old")
    ```

    Working with CSV data:
    ```python
    import csv
    from io import StringIO
    from agentle.parsing.page_item.table_page_item import TablePageItem

    # Create a table
    table = TablePageItem(
        rows=[
            ["Product", "Price", "Quantity"],
            ["Widget A", "$10.99", "5"],
            ["Widget B", "$24.99", "2"],
            ["Widget C", "$5.49", "10"]
        ],
        csv="Product,Price,Quantity\nWidget A,$10.99,5\nWidget B,$24.99,2\nWidget C,$5.49,10"
    )

    # Use the CSV data with the csv module
    csv_reader = csv.reader(StringIO(table.csv))
    headers = next(csv_reader)  # Get headers

    # Calculate total value
    total = 0
    for row in csv_reader:
        product, price, quantity = row
        price_value = float(price.replace('$', ''))
        quantity_value = int(quantity)
        row_total = price_value * quantity_value
        total += row_total
        print(f"{product}: {quantity} × {price} = ${row_total:.2f}")

    print(f"Total value: ${total:.2f}")
    ```

    Adding a table to a section:
    ```python
    from agentle.parsing.section_content import SectionContent
    from agentle.parsing.page_item.heading_page_item import HeadingPageItem
    from agentle.parsing.page_item.text_page_item import TextPageItem
    from agentle.parsing.page_item.table_page_item import TablePageItem

    # Create a heading and text intro
    heading = HeadingPageItem(heading="Quarterly Results", lvl=2)
    intro = TextPageItem(text="The following table shows our quarterly results:")

    # Create a table with the data
    table = TablePageItem(
        rows=[
            ["Quarter", "Revenue", "Expenses", "Profit"],
            ["Q1", "$1.2M", "$0.8M", "$0.4M"],
            ["Q2", "$1.5M", "$0.9M", "$0.6M"],
            ["Q3", "$1.8M", "$1.0M", "$0.8M"],
            ["Q4", "$2.0M", "$1.2M", "$0.8M"]
        ],
        csv="Quarter,Revenue,Expenses,Profit\nQ1,$1.2M,$0.8M,$0.4M\nQ2,$1.5M,$0.9M,$0.6M\nQ3,$1.8M,$1.0M,$0.8M\nQ4,$2.0M,$1.2M,$0.8M",
        is_perfect_table=True
    )

    # Create a section with all components
    section = SectionContent(
        number=1,
        text="Quarterly Results\n\nThe following table shows our quarterly results:\n\n[Table with quarterly results]",
        items=[heading, intro, table]
    )
    ```
    """

    type: Literal["table"] = Field(default="table")

    rows: Sequence[Sequence[str]] = Field(
        description="Rows of the table.",
    )

    csv: str = Field(
        description="CSV representation of the table",
    )

    is_perfect_table: bool = Field(
        default=False,
        description="Whether the table is a perfect table. A perfect table is a table that is well-structured and has no irregularities.",
    )

    model_config = ConfigDict(frozen=True)
