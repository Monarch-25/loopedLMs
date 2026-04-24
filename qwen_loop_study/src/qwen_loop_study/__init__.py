"""Qwen loop-vs-dense study package."""

from .compat import configure_transformers_runtime

configure_transformers_runtime()

__all__ = ["configure_transformers_runtime"]
