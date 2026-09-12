"""Canonicalization and content hashing.

Every result file is addressed by a config hash and carries a run fingerprint.
Those two values decide whether a number measured today may be compared against
one measured last month, so the encoding here is deliberately rigid: change it
and every committed result is orphaned.

blake2b rather than sha256 purely because it takes a `digest_size`, which lets
a config hash be short enough to sit in a filename without truncation games.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import blake2b
from typing import Any, Final

#: 8 bytes -> 16 hex characters. Short enough to read in a filename; a
#: collision across the handful of configs a project compares is not credible.
CONFIG_DIGEST_SIZE: Final[int] = 8

#: 16 bytes -> 32 hex characters. Run fingerprints are never eyeballed, so
#: there is no reason to economise.
FINGERPRINT_DIGEST_SIZE: Final[int] = 16


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a payload to its one canonical JSON form.

    Three choices, each load-bearing:

    - ``sort_keys=True`` -- mapping key order is an artifact of how a dict was
      built, never meaning, so it must not reach the hash. Nested mappings are
      sorted too, which ``json`` does at every depth.
    - ``separators=(",", ":")`` -- whitespace carries no meaning and would
      otherwise change the hash.
    - ``ensure_ascii=False`` -- escaping would make the canonical form depend
      on the encoder rather than the content.

    List order is deliberately preserved. It is meaningful: it is the order
    stages run in.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(payload: Mapping[str, Any], *, digest_size: int = CONFIG_DIGEST_SIZE) -> str:
    """Hash a payload by its content, independent of how it was constructed."""
    encoded = canonical_json(payload).encode("utf-8")
    return blake2b(encoded, digest_size=digest_size).hexdigest()


def run_fingerprint(
    *,
    config_hash: str,
    golden_set_hash: str,
    corpus_manifest_hash: str,
    code_version: str,
) -> str:
    """Identify the exact conditions that produced a result.

    A config hash alone is not enough to make two numbers comparable. There are
    four independent ways for conditions to drift, and only one of them is the
    configuration:

    1. the configuration changed
    2. the golden set changed -- queries added, removed, or re-labeled
    3. the corpus changed, or was re-normalized so character offsets moved
    4. the code changed, so a scorer fix legitimately moved every number

    Recording only (1) is how an evaluation harness silently lies: a metric
    "improves" because someone deleted the queries it was failing.

    The components are hashed as a JSON object rather than concatenated, so
    they cannot be positionally confused -- plain concatenation would make
    ``("ab", "c")`` and ``("a", "bc")`` collide.
    """
    return content_hash(
        {
            "config_hash": config_hash,
            "golden_set_hash": golden_set_hash,
            "corpus_manifest_hash": corpus_manifest_hash,
            "code_version": code_version,
        },
        digest_size=FINGERPRINT_DIGEST_SIZE,
    )
