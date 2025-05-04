from collections.abc import Callable, MutableMapping

from agentle.parsing.document_parser import DocumentParser

_parser_registry: MutableMapping[str, type[DocumentParser]] = {}


def parses[ParserT: DocumentParser](
    *extensions: str,
) -> Callable[[type[ParserT]], type[ParserT]]:
    """Decorator to register DocumentParser subclasses for specific file extensions."""

    def decorator(
        parser_cls: type[ParserT],
    ) -> type[ParserT]:
        for extension in extensions:
            _parser_registry[extension] = parser_cls
        return parser_cls

    return decorator
