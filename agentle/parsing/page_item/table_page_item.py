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

    **Inheritance:**

    *   Inherits from `PageItem`.

    **Attributes:**

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

    *   `md` (str):
        Inherited from `PageItem`. Markdown representation of the table.
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
