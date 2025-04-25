from __future__ import annotations

from dataclasses import dataclass

from rsb.decorators.value_objects import valueobject


@dataclass(frozen=True)
@valueobject
class Blob:
    data: bytes
    extension: str
