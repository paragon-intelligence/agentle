from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, MutableSequence, Sequence
from functools import cached_property
from itertools import chain
from typing import Any

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.parsing.chunk import Chunk
from agentle.parsing.chunking.chunking_config import ChunkingConfig
from agentle.parsing.chunking.chunking_strategy import ChunkingStrategy
from agentle.parsing.section_content import SectionContent


class ParsedFile(BaseModel):
    """
    Represents a fully parsed document with its sections and metadata.

    The ParsedFile class is the main output of the document parsing process.
    It contains the document's name and a collection of sections representing
    the content of the document. This structured representation makes it easy
    to work with parsed content from any supported file type in a consistent way.

    **Attributes:**

    *   `name` (str):
        The name of the document, typically derived from the original file name.

        **Example:**
        ```python
        doc = ParsedFile(name="report.pdf", sections=[])
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

        doc = ParsedFile(name="document.txt", sections=[section1, section2])

        for section in doc.sections:
            print(f"Section {section.number}: {section.text[:20]}...")
        ```

    **Usage Examples:**

    Creating a ParsedFile with multiple sections:
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
    doc = ParsedFile(
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

    metadata: Mapping[str, Any] = Field(
        default_factory=dict, description="Additional metadata of the document."
    )

    def split(self) -> Sequence[Chunk]: ...

    @cached_property
    def unique_id(self) -> str:
        """
        Generates a deterministic, unique ID for the file based on its name and content.

        This ID is stable across multiple runs, ensuring that the same file content
        always results in the same unique_id.

        Returns:
            str: A unique and deterministic identifier.
        """
        # Sanitize name for a clean base identifier
        base_name = self._sanitize_name()

        # Create a single, deterministic string from all section texts.
        # This avoids the non-deterministic behavior of Python's built-in hash().
        content_text = "".join(s.text for s in self.sections)

        # Combine the filename and the full content text to create a unique fingerprint.
        # This ensures that files with the same content but different names have different IDs.
        identifier_string = f"{self.name}:{content_text}"

        # Generate a deterministic UUIDv5. UUIDv5 uses SHA-1 hashing, which is
        # deterministic and will always produce the same result for the same input string.
        namespace = uuid.NAMESPACE_OID
        content_uuid = uuid.uuid5(namespace, identifier_string)

        return f"{base_name}_{str(content_uuid)[:8]}"

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

            doc = ParsedFile(
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

    def chunkify(
        self,
        strategy: ChunkingStrategy,
        generation_provider: GenerationProviderType | None = None,
    ) -> Sequence[Chunk]:
        return run_sync(
            self.chunkify_async,
            strategy=strategy,
            generation_provider=generation_provider,
        )

    async def chunkify_async(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        generation_provider: GenerationProviderType | None = None,
        config: ChunkingConfig | None = None,
    ) -> Sequence[Chunk]:
        match strategy:
            case ChunkingStrategy.AUTO:
                if generation_provider is None:
                    raise ValueError(
                        'Instance of GenerationProviderType needs to be passed if strategy is "auto"'
                    )
                return await self.chunkify_async(
                    strategy=ChunkingStrategy.RECURSIVE_CHARACTER
                )  # TODO(arthur)
            case ChunkingStrategy.RECURSIVE_CHARACTER:
                markdown = self.md
                return self._recursive_character_split(markdown, config)

        return []

    def _recursive_character_split(
        self,
        text: str,
        config: ChunkingConfig | None = None,
    ) -> Sequence[Chunk]:
        """
        Split text using recursive character splitting strategy.

        This method tries to split text at natural boundaries in order of preference:
        1. Double newlines (paragraphs)
        2. Single newlines (lines)
        3. Sentences (periods, exclamation marks, question marks)
        4. Spaces (words)
        5. Characters (last resort)

        Args:
            text: The text to split
            config: Configuration for chunking (chunk_size, chunk_overlap)

        Returns:
            Sequence of Chunk objects
        """
        # Default configuration
        chunk_size = config.get("chunk_size", 1000) if config else 1000
        chunk_overlap = config.get("chunk_overlap", 200) if config else 200

        # Validate configuration
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        # Define separators in order of preference (most natural to least natural)
        separators: Sequence[str] = [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences ending with period
            "! ",  # Sentences ending with exclamation
            "? ",  # Sentences ending with question
            " ",  # Words
            "",  # Characters (fallback)
        ]

        chunks = []
        current_chunks = [text]

        for separator in separators:
            next_chunks: MutableSequence[str] = []

            for chunk in current_chunks:
                if len(chunk) <= chunk_size:
                    # Chunk is already small enough
                    next_chunks.append(chunk)
                else:
                    # Split this chunk further
                    if separator == "":
                        # Last resort: split by characters
                        split_chunks = self._split_by_characters(
                            chunk, chunk_size, chunk_overlap
                        )
                    else:
                        split_chunks = self._split_by_separator(
                            chunk, separator, chunk_size, chunk_overlap
                        )
                    next_chunks.extend(split_chunks)

            current_chunks = next_chunks

            # Check if all chunks are now small enough
            if all(len(chunk) <= chunk_size for chunk in current_chunks):
                break

        # Create Chunk objects with metadata
        for i, chunk_text in enumerate(current_chunks):
            if chunk_text.strip():  # Only create chunks with non-empty content
                chunk_metadata = {
                    "source_document_id": self.unique_id,
                    "source_document": self.name,
                    "chunk_index": i,
                    "chunk_size": len(chunk_text),
                    "chunking_strategy": "recursive_character",
                    "original_document_metadata": self.metadata,
                    "total_chunks": len([c for c in current_chunks if c.strip()]),
                }

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text.strip(),
                        metadata=chunk_metadata,
                    )
                )

        return chunks

    def _split_by_separator(
        self,
        text: str,
        separator: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Split text by a specific separator while respecting chunk size and overlap."""
        parts = text.split(separator)
        chunks: MutableSequence[str] = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # Add separator back (except for the last part)
            if i < len(parts) - 1:
                part_with_sep = part + separator
            else:
                part_with_sep = part

            # Check if adding this part would exceed chunk_size
            if current_chunk and len(current_chunk) + len(part_with_sep) > chunk_size:
                # Save current chunk and start a new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap if possible
                if chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + part_with_sep
                else:
                    current_chunk = part_with_sep
            else:
                current_chunk += part_with_sep

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_characters(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Split text by characters as a last resort."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            # Move start position considering overlap
            start = max(start + chunk_size - chunk_overlap, start + 1)

        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters from text for overlap."""
        if len(text) <= overlap_size:
            return text

        # Try to find a good breaking point for overlap (prefer word boundaries)
        overlap_text = text[-overlap_size:]

        # Try to start at a word boundary
        space_index = overlap_text.find(" ")
        if space_index > 0:
            overlap_text = overlap_text[space_index + 1 :]

        return overlap_text

    def merge_all(self, others: Sequence[ParsedFile]) -> ParsedFile:
        """
        Merge this document with a sequence of other ParsedFile objects.

        This method combines the current document with other ParsedFile objects,
        keeping the name of the current document but merging all sections from all documents.

        Args:
            others (Sequence[ParsedFile]): Other parsed documents to merge with this one

        Returns:
            ParsedFile: A new document containing all sections from this document
                           and the other documents

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            # Create sample documents
            doc1 = ParsedFile(
                name="doc1.txt",
                sections=[SectionContent(number=1, text="Content from doc1")]
            )

            doc2 = ParsedFile(
                name="doc2.txt",
                sections=[SectionContent(number=1, text="Content from doc2")]
            )

            doc3 = ParsedFile(
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

        # Merge all metadata dicts, with self.metadata taking precedence over others
        merged_metadata: Mapping[str, Any] = {}
        for other in others:
            merged_metadata.update(other.metadata)

        merged_metadata.update(self.metadata)

        return ParsedFile(
            name=self.name,
            sections=list(chain(self.sections, *[other.sections for other in others])),
            metadata=merged_metadata,
        )

    @classmethod
    def from_sections(
        cls,
        name: str,
        sections: Sequence[SectionContent],
        metadata: Mapping[str, Any] | None = None,
    ) -> ParsedFile:
        """
        Create a ParsedFile from a name and a sequence of sections.

        This factory method provides a convenient way to create a ParsedFile
        by specifying the document name and its sections.

        Args:
            name (str): The name to give to the document
            sections (Sequence[SectionContent]): The sections to include in the document

        Returns:
            ParsedFile: A new ParsedFile instance with the specified name and sections

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            sections = [
                SectionContent(number=1, text="First section"),
                SectionContent(number=2, text="Second section"),
                SectionContent(number=3, text="Third section")
            ]

            doc = ParsedFile.from_sections("compiled_document.txt", sections)

            print(doc.name)  # Output: compiled_document.txt
            print(len(doc.sections))  # Output: 3
            ```
        """
        return cls(name=name, sections=sections, metadata=metadata or {})

    @classmethod
    def from_parsed_files(cls, files: Sequence[ParsedFile]) -> ParsedFile:
        """
        Create a merged ParsedFile from multiple existing ParsedFile objects.

        This factory method provides a convenient way to combine multiple documents
        into a single document. The resulting document will have the name "MergedFile"
        and will contain all sections from all input files.

        Args:
            files (Sequence[ParsedFile]): The ParsedFile objects to merge

        Returns:
            ParsedFile: A new ParsedFile containing all sections from the input files

        Example:
            ```python
            from agentle.parsing.section_content import SectionContent

            # Create sample documents
            doc1 = ParsedFile(
                name="chapter1.txt",
                sections=[SectionContent(number=1, text="Chapter 1 content")]
            )

            doc2 = ParsedFile(
                name="chapter2.txt",
                sections=[SectionContent(number=1, text="Chapter 2 content")]
            )

            # Merge documents
            book = ParsedFile.from_parsed_files([doc1, doc2])

            print(book.name)  # Output: MergedFile
            print(len(book.sections))  # Output: 2
            ```
        """
        # Merge all sections
        merged_sections = list(chain(*[file.sections for file in files]))

        # Merge metadata from all files (later files override earlier ones on key conflict)
        merged_metadata: Mapping[str, Any] = {}
        for file in files:
            merged_metadata.update(file.metadata)

        return cls(
            name="MergedFile",
            sections=merged_sections,
            metadata=merged_metadata,
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

            doc = ParsedFile(
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

    def _sanitize_name(self) -> str:
        """Extract name sanitization into reusable method."""
        if not self.name:
            return "unnamed_file"

        # Remove file extension
        base_name = self.name.rsplit(".", 1)[0] if "." in self.name else self.name

        # Convert to lowercase and replace non-alphanumeric characters with underscores
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name.lower())

        # Remove consecutive underscores and strip leading/trailing underscores
        base_name = re.sub(r"_+", "_", base_name).strip("_")

        # Ensure it doesn't start with a number
        if base_name and base_name[0].isdigit():
            base_name = f"file_{base_name}"

        return base_name if base_name else "unnamed_file"


# strategy: Literal[
#             # Intelligent Strategy Selection
#             "auto",  # LLM-powered automatic strategy selection
#             # Semantic & AI-Powered Approaches
#             "semantic_chunking",  # Embedding-based semantic similarity
#             "contextual",  # AI-powered contextual chunking (Anthropic-style)
#             "late_chunking",  # Query-time dynamic chunking
#             # Structure-Aware Strategies
#             "recursive_character",  # Hierarchical separator-based (most common)
#             "document_structure",  # Header/section-based chunking
#             "hierarchical",  # Multi-level document structure
#             # Content-Type Specific
#             "code_aware",  # Syntax-preserving for code documents
#             "markdown_aware",  # Markdown structure preservation
#             "html_aware",  # HTML tag-based chunking
#             # Basic Splitting Methods
#             "fixed_size",  # Fixed character/token count
#             "sentence_based",  # Split by sentence boundaries
#             "paragraph_based",  # Split by paragraph boundaries
#             "token_based",  # Split by token count
#             # Advanced Techniques
#             "sliding_window",  # Overlapping window approach
#             "hybrid",  # Multiple strategy combination
#             "adaptive",  # Content-complexity adaptive sizing
#             # Domain-Specific
#             "academic_paper",  # Research paper structure-aware
#             "legal_document",  # Legal clause-aware chunking
#             "business_report",  # Business document structure
#             "technical_manual",  # Technical documentation optimized
#             # Performance Optimized
#             "fast_fixed",  # Optimized for speed
#             "balanced",  # Speed-accuracy balance
#             "high_quality",  # Accuracy-optimized regardless of speed
#         ],
