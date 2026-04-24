"""Minimal string-keyed registry used by dataset / model / head factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class Registry(Generic[T]):
    """Minimal registry: `@reg.register("name")` to decorate; `reg.get("name")` to look up."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def _deco(fn: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise KeyError(f"duplicate {self._kind} registration: {name!r}")
            self._items[name] = fn
            return fn

        return _deco

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._items:
            raise KeyError(f"unknown {self._kind}: {name!r}; known={sorted(self._items)}")
        return self._items[name]

    def keys(self) -> list[str]:
        return sorted(self._items)
