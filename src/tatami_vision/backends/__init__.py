"""Registro de backends de clasificación."""

from __future__ import annotations

from .base import BackendStatus, BlockResult, PositionClassifier

BACKEND_NAMES = ("ollama", "anthropic", "local", "mock")


def _cls(name: str) -> type[PositionClassifier]:
    if name == "ollama":
        from .ollama import OllamaClassifier

        return OllamaClassifier
    if name == "anthropic":
        from .anthropic import AnthropicClassifier

        return AnthropicClassifier
    if name == "local":
        from .local import LocalClassifier

        return LocalClassifier
    if name == "mock":
        from .mock import MockClassifier

        return MockClassifier
    raise ValueError(f"backend desconocido: {name!r}; opciones: {BACKEND_NAMES}")


def get_backend(name: str, model: str | None = None, **opts) -> PositionClassifier:
    return _cls(name)(model=model, **opts)


def backend_statuses() -> list[BackendStatus]:
    return [_cls(n).status() for n in BACKEND_NAMES]


__all__ = [
    "BACKEND_NAMES",
    "BackendStatus",
    "BlockResult",
    "PositionClassifier",
    "backend_statuses",
    "get_backend",
]
