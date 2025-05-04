from collections.abc import Callable, MutableMapping
from typing import TypeVar, Type

from agentle.parsing.document_parser import DocumentParser

_parser_registry: MutableMapping[str, type[DocumentParser]] = {}

# Define a TypeVar constrained to DocumentParser
ParserT = TypeVar("ParserT", bound=DocumentParser)


def parses(
    *extensions: str,
) -> Callable[[Type[ParserT]], Type[ParserT]]:
    """Decorator to register DocumentParser subclasses for specific file extensions."""

    def decorator(
        parser_cls: Type[ParserT],
    ) -> Type[ParserT]:
        # No need for the inner class '_' or functools.wraps as we return the original class.
        for extension in extensions:
            _parser_registry[extension] = parser_cls
        return parser_cls

    return decorator
