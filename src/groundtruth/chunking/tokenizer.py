"""Loading the vendored tokenizer.

The tokenizer is committed to the repository rather than downloaded, for the
same reason the embedding cache is (ADR-0003): chunk boundaries must be
reproducible with no network access. A tokenizer fetched at runtime could
change under a committed baseline and silently move every chunk boundary,
which would move every metric.

Only ``tokenizer.json`` is required -- the Rust ``tokenizers`` package reads it
directly, with no torch and no ``transformers`` dependency.
"""

from __future__ import annotations

import functools
from pathlib import Path

from tokenizers import Tokenizer

from groundtruth.paths import tokenizers_dir

TOKENIZER_FILENAME = "tokenizer.json"


class TokenizerNotVendoredError(Exception):
    """The tokenizer a configuration asks for is not committed to the repo."""


def tokenizer_slug(tokenizer_id: str) -> str:
    """Map a HuggingFace id to its vendored directory name.

    ``BAAI/bge-small-en-v1.5`` -> ``bge-small-en-v1.5``. The organisation
    prefix is dropped because it would otherwise create a nested directory for
    no benefit.
    """
    return tokenizer_id.rsplit("/", maxsplit=1)[-1]


def tokenizer_path(tokenizer_id: str) -> Path:
    return tokenizers_dir() / tokenizer_slug(tokenizer_id) / TOKENIZER_FILENAME


@functools.lru_cache(maxsize=8)
def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    """Load a vendored tokenizer.

    Cached because chunking a corpus loads the same tokenizer once per
    document otherwise, and parsing a 700 KB vocabulary repeatedly dominates
    the runtime of an operation that should be trivial.
    """
    path = tokenizer_path(tokenizer_id)
    if not path.is_file():
        raise TokenizerNotVendoredError(
            f"tokenizer {tokenizer_id!r} is not vendored: expected {path}. "
            f"Chunking must be reproducible offline, so tokenizers are "
            f"committed rather than downloaded. Add it under "
            f"{tokenizers_dir()} with a MANIFEST.json recording its revision."
        )
    return Tokenizer.from_file(str(path))
