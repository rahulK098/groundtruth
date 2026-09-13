"""Embedding cache key derivation.

The key covers **the exact string sent to the model**, not the chunk it came
from. That is what makes prefix handling correct for free: bge instructs
queries with a prefix but leaves passages bare, so changing the prefix
invalidates query vectors and leaves passage vectors alone, with no separate
"is this a query" flag to get wrong.

Model identity is part of the key, revision included. Without that, upgrading
weights would silently serve vectors produced by the old ones.
"""

from __future__ import annotations

import hashlib

from groundtruth.config.hashing import content_hash

#: 8 bytes -> 16 hex characters, matching the config hash. Collisions across a
#: few tens of thousands of chunks are not credible.
_KEY_DIGEST_SIZE = 8


def embedding_key(text: str, *, model_id: str, revision: str, normalize: bool) -> str:
    """Content address of one embedding.

    The text is pre-hashed rather than embedded in the payload so the key
    derivation cost does not scale with document length, and so the payload
    stays a small fixed-size object.
    """
    return content_hash(
        {
            "model_id": model_id,
            "revision": revision,
            "normalize": normalize,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        digest_size=_KEY_DIGEST_SIZE,
    )


def text_digest(text: str) -> str:
    """sha256 of an embedded string, recorded per row for auditability."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
