"""The embedder interface.

A Protocol rather than a base class so the real model, the cache guards, and
the test fake are interchangeable without any of them importing the others --
which is what keeps ``sentence_transformers`` out of the gate's import graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns strings into vectors.

    Implementations embed **exactly the strings given**. Instruction prefixes
    are applied by the caller, because the cache key covers the final string
    (see ``hashing.py``) and an embedder that silently rewrote its input would
    break that correspondence.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalize(self) -> bool: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dimension)`` float32 array, row-aligned."""
        ...
