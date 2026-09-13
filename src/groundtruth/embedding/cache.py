"""Cache-backed embedders.

Two implementations, and the difference between them is the point.

``CachedEmbedder`` serves what it has and computes the rest. Used when
regenerating the cache.

``CachedOnlyEmbedder`` **raises on a miss**. Used by the gate, and by anything
claiming to be reproducible.

The second one exists because the failure it prevents is the worst kind
available to this project. Without it, a stale cache degrades into "some
vectors are current, some are not" and the harness reports plausible,
confident, wrong numbers -- with nothing anywhere to indicate a problem. A
loud failure naming the exact regeneration command is strictly better.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.protocol import Embedder
from groundtruth.embedding.store import LoadedStore


class EmbeddingCacheMissError(Exception):
    """The committed cache does not cover a requested string."""


class CachedOnlyEmbedder:
    """Serves committed vectors and refuses to compute anything.

    This is what makes the gate honest: it cannot download a model, cannot
    reach the network, and cannot quietly mix fresh vectors with stale ones.
    """

    def __init__(self, store: LoadedStore) -> None:
        self._store = store
        self._model_id = str(store.meta["model_id"])
        self._revision = str(store.meta["revision"])
        self._normalize = bool(store.meta["normalize"])
        self._dimension = int(store.meta["dimension"])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        keys = [
            embedding_key(
                text,
                model_id=self._model_id,
                revision=self._revision,
                normalize=self._normalize,
            )
            for text in texts
        ]
        missing = [key for key in keys if key not in self._store]

        if missing:
            raise EmbeddingCacheMissError(
                f"{len(missing)} of {len(keys)} strings are not in the committed "
                f"embedding cache for {self._model_id}@{self._revision[:12]}.\n\n"
                f"This usually means a configuration changed (chunk size, "
                f"embedding model, or the query prefix) without regenerating "
                f"the cache. Regenerate and commit it:\n\n"
                f"    uv sync --frozen --extra dev --extra models\n"
                f"    uv run gt cache build\n"
                f"    git add data/cache && git commit -m 'chore: regenerate embedding cache'\n\n"
                f"The cache is deliberately authoritative here: computing the "
                f"missing vectors on the fly would mix fresh and stale "
                f"embeddings and report plausible, wrong numbers."
            )

        return np.stack([self._store.vector_for(key) for key in keys])


class CachedEmbedder:
    """Serves what the cache has and delegates the rest.

    Used when building or extending a cache, never by the gate.
    """

    def __init__(self, store: LoadedStore | None, delegate: Embedder) -> None:
        self._store = store
        self._delegate = delegate
        #: Vectors computed this session, to be written back.
        self.computed: dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    @property
    def revision(self) -> str:
        return self._delegate.revision

    @property
    def dimension(self) -> int:
        return self._delegate.dimension

    @property
    def normalize(self) -> bool:
        return self._delegate.normalize

    def _key(self, text: str) -> str:
        return embedding_key(
            text,
            model_id=self.model_id,
            revision=self.revision,
            normalize=self.normalize,
        )

    def _lookup(self, key: str) -> np.ndarray | None:
        """Vector for a key from this session or the committed store."""
        if key in self.computed:
            return self.computed[key]
        if self._store is not None and key in self._store:
            return self._store.vector_for(key)
        return None

    def _compute(
        self, texts: Sequence[str], keys: Sequence[str], rows: Sequence[int], out: np.ndarray
    ) -> None:
        """Embed the rows the cache could not serve, and record them."""
        # Deduplicate before calling the model: a legal corpus repeats
        # boilerplate constantly, and embedding the same string twice is pure
        # cost paid on every rebuild.
        unique: dict[str, list[int]] = {}
        for row in rows:
            unique.setdefault(keys[row], []).append(row)

        representatives = [texts[same_rows[0]] for same_rows in unique.values()]
        fresh = self._delegate.embed(representatives)

        for (key, same_rows), vector in zip(unique.items(), fresh, strict=True):
            self.computed[key] = vector
            for row in same_rows:
                out[row] = vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        keys = [self._key(text) for text in texts]
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)

        pending: list[int] = []
        for row, key in enumerate(keys):
            cached = self._lookup(key)
            if cached is None:
                pending.append(row)
            else:
                out[row] = cached

        if pending:
            self._compute(texts, keys, pending, out)

        return out
