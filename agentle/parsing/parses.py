from collections.abc import Callable, MutableMapping

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsers.html import HTMLParser

"""
Parser Registry and Extension Mapping System

This module provides a registry system that maps file extensions to their corresponding
document parser classes. It uses a decorator pattern to register parser classes with the
extensions they can handle.

The main components are:
1. `parser_registry`: A dictionary mapping file extensions to parser classes
2. `parses` decorator: A decorator function that registers parser classes for specific extensions

This system enables the automatic selection of the appropriate parser based on a file's extension,
which is central to the framework's ability to handle different file types transparently.
"""

parser_registry: MutableMapping[str, type[DocumentParser]] = {
    "html": HTMLParser,
    
}
"""
Global registry mapping file extensions to their respective DocumentParser classes.

This dictionary is populated by the `@parses` decorator. Each key is a file extension
(without the leading dot, e.g., "pdf", "docx"), and each value is the DocumentParser
subclass that can parse that file type.
"""


def parses[ParserT: DocumentParser](
    *extensions: str,
) -> Callable[[type[ParserT]], type[ParserT]]:
    """
    Decorator to register DocumentParser subclasses for specific file extensions.

    This decorator associates parser classes with the file extensions they can handle,
    allowing the framework to automatically select the appropriate parser for each file type.
    Extensions should be specified without the leading dot (e.g., "pdf", not ".pdf").

    Args:
        *extensions (str): One or more file extensions that this parser can handle.
            For example: "pdf", "docx", "txt".

    Returns:
        Callable: A decorator function that registers the parser class and returns it unmodified.

    Example:
        ```python
        from agentle.parsing.document_parser import DocumentParser
        from agentle.parsing.parses import parses

        @parses("txt", "text", "log")
        class TextFileParser(DocumentParser):
            async def parse_async(self, document_path: str) -> ParsedDocument:
                # Implementation for parsing text files
                ...
        ```

    Note:
        When multiple parsers are registered for the same extension, the last one
        registered will be used. This allows for overriding default parsers with
        custom implementations.
    """

    def decorator(
        parser_cls: type[ParserT],
    ) -> type[ParserT]:
        for extension in extensions:
            # Remove leading dot if present for consistency
            clean_ext = extension.lstrip(".")
            parser_registry[clean_ext] = parser_cls
        return parser_cls

    return decorator
