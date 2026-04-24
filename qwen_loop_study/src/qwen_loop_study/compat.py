"""Compatibility helpers for HF/TRL runtime imports."""

from __future__ import annotations

import os


def configure_transformers_runtime() -> None:
    """Disable TensorFlow imports so TRL stays usable in minimal envs."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
