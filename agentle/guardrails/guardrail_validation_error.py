from __future__ import annotations


class GuardrailValidationError(Exception):
    def raise_as_another_exception(self, other: type[Exception]) -> None:
        raise other(self)
