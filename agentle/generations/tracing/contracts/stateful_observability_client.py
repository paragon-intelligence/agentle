from __future__ import annotations

import abc
from datetime import datetime
from typing import Sequence


class StatefulObservabilityClient(abc.ABC):
    @abc.abstractmethod
    def trace(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient: ...

    @abc.abstractmethod
    def generation(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient: ...

    @abc.abstractmethod
    def span(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient: ...

    @abc.abstractmethod
    def event(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient: ...

    @abc.abstractmethod
    def end(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient: ...
