from collections.abc import Callable


from agentle.parsing.document_parser import DocumentParser

_parser_registry: dict[str, type[DocumentParser]] = {}


def parses(
    *extensions: str,
) -> Callable[[type[DocumentParser]], type[DocumentParser]]:
    """
    Decorator to register a file parser class for specific file extensions.

    This decorator is used to register a file parser class for specific file extensions.
    It adds the parser class to the global parser registry, allowing the `FileParser`
    to automatically select the correct parser based on the file extension. Should be
    used internally only by concrete intellibricks parser classes.

    **Parameters:**

    *   `extensions` (str): One or more file extensions that the parser class supports.

    **Returns:**

    *   `Callable[[type[FileParser]], type[FileParser]]`: A decorator function that registers the parser class.

    **Example:**

    ```python
    from intelliparse.parsers import FileParser, parses

    @_parses("txt")
    class CustomTxtFileParser(FileParser):
        async def parse_async(self, file: RawFile) -> ParsedFile:
            # Add parsing logic here
            pass
    ```
    """

    def decorator(
        parser_cls: type[DocumentParser],
    ) -> type[DocumentParser]:
        for extension in extensions:
            _parser_registry[extension] = parser_cls
        return parser_cls

    return decorator
