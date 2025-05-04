from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.parsing.section_content import SectionContent


class ParsedDocument(BaseModel):
    """
    Represents a fully parsed document with its sections and metadata.

    The ParsedDocument class is the main output of the document parsing process.
    It contains the document's name and a collection of sections representing
    the content of the document. This structured representation makes it easy
    to work with parsed content from any supported file type in a consistent way.

    **Attributes:**

    *   `name` (str):
        The name of the document, typically derived from the original file name.

        **Example:**
        ```python
        doc = ParsedDocument(name="report.pdf", sections=[])
        print(doc.name)  # Output: report.pdf
        ```

    *   `sections` (Sequence[SectionContent]):
        A sequence of SectionContent objects representing the document's content
        divided into logical sections or pages.

        **Example:**
        ```python
        from agentle.parsing.section_content import SectionContent

        section1 = SectionContent(number=1, text="First section content")
        section2 = SectionContent(number=2, text="Second section content")

        doc = ParsedDocument(name="document.txt", sections=[section1, section2])

        for section in doc.sections:
            print(f"Section {section.number}: {section.text[:20]}...")
        ```

    **Usage Examples:**

    Creating a ParsedDocument with multiple sections:
    ```python
    from agentle.parsing.section_content import SectionContent

    # Create sections
    intro = SectionContent(
        number=1,
        text="Introduction to the topic",
        md="# Introduction\n\nThis document covers..."
    )

    body = SectionContent(
        number=2,
        text="Main content of the document",
        md="## Main Content\n\nThe details of..."
    )

    conclusion = SectionContent(
        number=3,
        text="Conclusion of the document",
        md="## Conclusion\n\nIn summary..."
    )

    # Create the parsed document
    doc = ParsedDocument(
        name="example_document.docx",
        sections=[intro, body, conclusion]
    )

    # Access the document content
    print(f"Document: {doc.name}")
    print(f"Number of sections: {len(doc.sections)}")
    print(f"First section heading: {doc.sections[0].md.split('\\n')[0]}")
    ```
    """

    name: str = Field(
        description="Name of the file",
    )

    sections: Sequence[SectionContent] = Field(
        description="Pages of the document",
    )

    @property
    def llm_described_text(self) -> str:
        """
        Generate a description of the document suitable for LLM processing.

        This property formats the document content in a structured XML-like format
        that is optimized for large language models to understand the document's
        structure and content.

        Returns:
            str: A structured string representation of the document

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            doc = ParsedDocument(
                name="example.txt",
                sections=[
                    SectionContent(number=1, text="First section", md="# First section"),
                    SectionContent(number=2, text="Second section", md="# Second section")
                ]
            )

            llm_text = doc.llm_described_text
            print(llm_text)
            # Output:
            # <file>
            #
            # **name:** example.txt
            # **sections:** <section_0> # First section </section_0> <section_1> # Second section </section_1>
            #
            # </file>
            ```
        """
        sections = " ".join(
            [
                f"<section_{num}> {section.md} </section_{num}>"
                for num, section in enumerate(self.sections)
            ]
        )
        return f"<file>\n\n**name:** {self.name} \n**sections:** {sections}\n\n</file>"

    def merge_all(self, others: Sequence[ParsedDocument]) -> ParsedDocument:
        """
        Merge this document with a sequence of other ParsedDocument objects.

        This method combines the current document with other ParsedDocument objects,
        keeping the name of the current document but merging all sections from all documents.

        Args:
            others (Sequence[ParsedDocument]): Other parsed documents to merge with this one

        Returns:
            ParsedDocument: A new document containing all sections from this document
                           and the other documents

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            # Create sample documents
            doc1 = ParsedDocument(
                name="doc1.txt",
                sections=[SectionContent(number=1, text="Content from doc1")]
            )

            doc2 = ParsedDocument(
                name="doc2.txt",
                sections=[SectionContent(number=1, text="Content from doc2")]
            )

            doc3 = ParsedDocument(
                name="doc3.txt",
                sections=[SectionContent(number=1, text="Content from doc3")]
            )

            # Merge documents with doc1 as the base
            merged = doc1.merge_all([doc2, doc3])

            print(merged.name)  # Output: doc1.txt
            print(len(merged.sections))  # Output: 3
            ```
        """
        from itertools import chain

        return ParsedDocument(
            name=self.name,
            sections=list(chain(self.sections, *[other.sections for other in others])),
        )

    @classmethod
    def from_sections(
        cls, name: str, sections: Sequence[SectionContent]
    ) -> ParsedDocument:
        """
        Create a ParsedDocument from a name and a sequence of sections.

        This factory method provides a convenient way to create a ParsedDocument
        by specifying the document name and its sections.

        Args:
            name (str): The name to give to the document
            sections (Sequence[SectionContent]): The sections to include in the document

        Returns:
            ParsedDocument: A new ParsedDocument instance with the specified name and sections

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            sections = [
                SectionContent(number=1, text="First section"),
                SectionContent(number=2, text="Second section"),
                SectionContent(number=3, text="Third section")
            ]

            doc = ParsedDocument.from_sections("compiled_document.txt", sections)

            print(doc.name)  # Output: compiled_document.txt
            print(len(doc.sections))  # Output: 3
            ```
        """
        return cls(name=name, sections=sections)

    @classmethod
    def from_parsed_files(cls, files: Sequence[ParsedDocument]) -> ParsedDocument:
        """
        Create a merged ParsedDocument from multiple existing ParsedDocument objects.

        This factory method provides a convenient way to combine multiple documents
        into a single document. The resulting document will have the name "MergedFile"
        and will contain all sections from all input files.

        Args:
            files (Sequence[ParsedDocument]): The ParsedDocument objects to merge

        Returns:
            ParsedDocument: A new ParsedDocument containing all sections from the input files

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            # Create sample documents
            doc1 = ParsedDocument(
                name="chapter1.txt",
                sections=[SectionContent(number=1, text="Chapter 1 content")]
            )

            doc2 = ParsedDocument(
                name="chapter2.txt",
                sections=[SectionContent(number=1, text="Chapter 2 content")]
            )

            # Merge documents
            book = ParsedDocument.from_parsed_files([doc1, doc2])

            print(book.name)  # Output: MergedFile
            print(len(book.sections))  # Output: 2
            ```
        """
        return cls(
            name="MergedFile",
            sections=list(chain(*[file.sections for file in files])),
        )

    @property
    def md(self) -> str:
        """
        Generate a complete markdown representation of the document.

        This property combines the markdown content of all sections into a single
        markdown string, making it easy to get a complete markdown version of the
        document's content.

        Returns:
            str: The combined markdown content of all sections

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            doc = ParsedDocument(
                name="document.md",
                sections=[
                    SectionContent(number=1, text="First section", md="# First section\nContent"),
                    SectionContent(number=2, text="Second section", md="# Second section\nMore content")
                ]
            )

            markdown = doc.md
            print(markdown)
            # Output:
            # # First section
            # Content
            # # Second section
            # More content
            ```
        """
        return "\n".join([sec.md or "" for sec in self.sections])
