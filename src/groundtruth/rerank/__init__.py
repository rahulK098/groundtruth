"""Cross-encoder reranking, and the committed cache that lets the gate run it.

A cross-encoder scores a (query, passage) pair jointly, so -- unlike an
embedding -- there is no per-text artifact to reuse. ADR-0012 commits the
pair scores instead, with a cache-only reranker for the gate that raises on a
miss, exactly as ``CachedOnlyEmbedder`` does for vectors.
"""
