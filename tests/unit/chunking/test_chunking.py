"""Fixed-window chunking.

The invariant everything else depends on is::

    chunk.text == document.text[chunk.char_start:chunk.char_end]

Relevance labels are character spans (ADR-0002), and scoring resolves them by
overlap against chunk character spans. If that mapping is off by even one
character, every metric is quietly wrong in a way no other test would catch --
so it is property-tested against the real vendored tokenizer, not a fake.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from groundtruth.chunking.fixed_window import chunk_document, chunking_hash
from groundtruth.chunking.models import Chunk
from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.chunking.tokenizer import (
    TokenizerNotVendoredError,
    load_tokenizer,
    tokenizer_path,
    tokenizer_slug,
)
from groundtruth.config.models import ChunkingConfig
from groundtruth.corpus.models import Document
from groundtruth.corpus.normalize import normalize_text

TOKENIZER_ID = "BAAI/bge-small-en-v1.5"

SENTENCE = (
    "Summary judgment is appropriate when there is no genuine dispute as to "
    "any material fact and the movant is entitled to judgment as a matter of law. "
)


def chunking(chunk_size: int = 64, chunk_overlap: int = 16) -> ChunkingConfig:
    return ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer_id=TOKENIZER_ID,
    )


def document(text: str, doc_id: str = "cl-1") -> Document:
    return Document(doc_id=doc_id, text=normalize_text(text))


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TOKENIZER_ID)


class TestTokenizerLoading:
    def test_loads_the_vendored_tokenizer(self, tokenizer):
        assert tokenizer.get_vocab_size() > 0

    def test_slug_drops_the_organisation_prefix(self):
        assert tokenizer_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"

    def test_slug_leaves_a_bare_name_alone(self):
        assert tokenizer_slug("bge-small-en-v1.5") == "bge-small-en-v1.5"

    def test_vendored_path_exists(self):
        assert tokenizer_path(TOKENIZER_ID).is_file()

    def test_missing_tokenizer_names_the_expected_path(self):
        # The error has to be actionable: a bare KeyError here would send
        # someone hunting through HuggingFace docs instead of the repo.
        with pytest.raises(TokenizerNotVendoredError, match="not vendored"):
            load_tokenizer("nonexistent/tokenizer-that-is-not-here")

    def test_loading_is_cached(self, tokenizer):
        # Parsing a 700 KB vocabulary per document would dominate runtime.
        assert load_tokenizer(TOKENIZER_ID) is tokenizer


class TestSingleDocument:
    def test_short_document_yields_one_chunk(self, tokenizer):
        doc = document("A short opinion.")
        chunks = chunk_document(doc, chunking(), tokenizer)
        assert len(chunks) == 1
        assert chunks[0].text == doc.text

    def test_long_document_yields_several_chunks(self, tokenizer):
        doc = document(SENTENCE * 20)
        assert len(chunk_document(doc, chunking(), tokenizer)) > 1

    def test_empty_document_yields_no_chunks(self, tokenizer):
        assert chunk_document(document(""), chunking(), tokenizer) == ()

    def test_whitespace_only_document_yields_no_chunks(self, tokenizer):
        assert chunk_document(document("   \n\n  "), chunking(), tokenizer) == ()

    def test_no_chunk_exceeds_the_window(self, tokenizer):
        doc = document(SENTENCE * 20)
        cfg = chunking()
        for chunk in chunk_document(doc, cfg, tokenizer):
            assert chunk.token_count <= cfg.chunk_size

    def test_chunks_cover_the_whole_document(self, tokenizer):
        # The last chunk must reach the end, or the tail of every opinion is
        # unretrievable and no label pointing there could ever be satisfied.
        doc = document(SENTENCE * 20)
        chunks = chunk_document(doc, chunking(), tokenizer)
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(doc.text.rstrip())

    def test_consecutive_chunks_overlap(self, tokenizer):
        doc = document(SENTENCE * 20)
        cfg = chunking(chunk_size=64, chunk_overlap=16)
        chunks = chunk_document(doc, cfg, tokenizer)
        for earlier, later in pairwise(chunks):
            assert later.token_start < earlier.token_end

    def test_stride_matches_the_configured_overlap(self, tokenizer):
        doc = document(SENTENCE * 20)
        cfg = chunking(chunk_size=64, chunk_overlap=16)
        chunks = chunk_document(doc, cfg, tokenizer)
        strides = {b.token_start - a.token_start for a, b in pairwise(chunks)}
        assert strides == {cfg.chunk_size - cfg.chunk_overlap}

    def test_zero_overlap_produces_disjoint_windows(self, tokenizer):
        doc = document(SENTENCE * 20)
        chunks = chunk_document(doc, chunking(chunk_size=64, chunk_overlap=0), tokenizer)
        for earlier, later in pairwise(chunks):
            assert later.token_start == earlier.token_end

    def test_does_not_emit_a_redundant_tail_chunk(self, tokenizer):
        # A final window wholly contained in its predecessor is pure cost: it
        # is embedded, stored, and can only ever duplicate a hit.
        doc = document(SENTENCE * 20)
        chunks = chunk_document(doc, chunking(), tokenizer)
        if len(chunks) > 1:
            assert chunks[-1].token_end > chunks[-2].token_end


class TestChunkIdentity:
    def test_ids_are_deterministic(self, tokenizer):
        doc = document(SENTENCE * 5)
        first = chunk_document(doc, chunking(), tokenizer)
        second = chunk_document(doc, chunking(), tokenizer)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_ids_are_unique_within_a_document(self, tokenizer):
        chunks = chunk_document(document(SENTENCE * 20), chunking(), tokenizer)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_ids_differ_across_chunking_configs(self, tokenizer):
        # Several chunk sizes are indexed over the same corpus at once, so a
        # collision between configs would cross-contaminate retrieval.
        doc = document(SENTENCE * 20)
        a = {c.chunk_id for c in chunk_document(doc, chunking(64, 16), tokenizer)}
        b = {c.chunk_id for c in chunk_document(doc, chunking(128, 16), tokenizer)}
        assert a.isdisjoint(b)

    def test_ids_differ_across_documents(self, tokenizer):
        cfg = chunking()
        a = chunk_document(document(SENTENCE * 5, "cl-1"), cfg, tokenizer)
        b = chunk_document(document(SENTENCE * 5, "cl-2"), cfg, tokenizer)
        assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})

    def test_chunking_hash_ignores_nothing_substantive(self):
        assert chunking_hash(chunking(64, 16)) != chunking_hash(chunking(64, 32))
        assert chunking_hash(chunking(64, 16)) == chunking_hash(chunking(64, 16))


class TestCharacterSpanInvariant:
    """The mapping that makes span-based relevance labels work at all."""

    def test_text_matches_the_claimed_span(self, tokenizer):
        doc = document(SENTENCE * 20)
        for chunk in chunk_document(doc, chunking(), tokenizer):
            assert chunk.text == doc.text[chunk.char_start : chunk.char_end]

    @given(st.text(min_size=1, max_size=1200))
    @settings(
        max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_holds_for_arbitrary_text(self, raw: str):
        doc = document(raw)
        if not doc.text:
            return
        tok = load_tokenizer(TOKENIZER_ID)
        for chunk in chunk_document(doc, chunking(), tok):
            assert chunk.text == doc.text[chunk.char_start : chunk.char_end]
            assert chunk.char_start < chunk.char_end

    @given(st.text(min_size=1, max_size=1200))
    @settings(
        max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_spans_are_monotonic(self, raw: str):
        doc = document(raw)
        if not doc.text:
            return
        chunks = chunk_document(doc, chunking(), load_tokenizer(TOKENIZER_ID))
        for earlier, later in pairwise(chunks):
            assert later.char_start >= earlier.char_start
            assert later.char_end > earlier.char_end


class TestCorpus:
    def test_chunks_every_document(self, tokenizer):
        docs = [document(SENTENCE * 3, f"cl-{i}") for i in range(4)]
        chunks = chunk_corpus(docs, chunking())
        assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}

    def test_output_order_is_independent_of_input_order(self, tokenizer):
        # Determinism here is what lets the embedding cache be committed.
        docs = [document(SENTENCE * 3, f"cl-{i}") for i in range(5)]
        forward = [c.chunk_id for c in chunk_corpus(docs, chunking())]
        reverse = [c.chunk_id for c in chunk_corpus(list(reversed(docs)), chunking())]
        assert forward == reverse

    def test_empty_corpus_yields_no_chunks(self):
        assert chunk_corpus([], chunking()) == ()


class _StubEncoding:
    def __init__(self, offsets):
        self.offsets = offsets


class _StubTokenizer:
    """Emits offsets chosen by the test, including degenerate ones.

    The real tokenizer does not produce zero-width offsets for ordinary tokens,
    so the defensive branch that skips them cannot be reached with real input.
    It is still worth proving it degrades to "skip" rather than raising -- an
    untested defensive branch is a guess, not a guarantee.
    """

    def __init__(self, offsets):
        self._offsets = offsets

    def encode(self, text, add_special_tokens=True):
        return _StubEncoding(self._offsets)


class TestDegenerateOffsets:
    def test_all_zero_width_offsets_yield_no_chunks(self):
        doc = document("some text here that is long enough")
        stub = _StubTokenizer([(0, 0)] * 8)
        assert chunk_document(doc, chunking(), stub) == ()

    def test_a_degenerate_leading_window_is_skipped_not_fatal(self):
        # First window is all zero-width, second is real. The real one must
        # still be emitted.
        doc = document("x" * 40)
        offsets = [(0, 0)] * 64 + [(10, 20)] * 64
        chunks = chunk_document(doc, chunking(64, 0), _StubTokenizer(offsets))
        assert len(chunks) == 1
        assert (chunks[0].char_start, chunks[0].char_end) == (10, 20)


class TestChunkValidation:
    """Chunk refuses to exist in a state that would corrupt span resolution."""

    def valid(self, **overrides):
        base = {
            "chunk_id": "abc123",
            "doc_id": "cl-1",
            "text": "hello",
            "char_start": 10,
            "char_end": 15,
            "token_start": 0,
            "token_end": 2,
        }
        return {**base, **overrides}

    def test_accepts_a_well_formed_chunk(self):
        assert Chunk.model_validate(self.valid()).token_count == 2

    def test_rejects_an_empty_character_span(self):
        with pytest.raises(ValidationError, match="char_end"):
            Chunk.model_validate(self.valid(char_end=10))

    def test_rejects_an_inverted_character_span(self):
        with pytest.raises(ValidationError, match="char_end"):
            Chunk.model_validate(self.valid(char_start=15, char_end=10))

    def test_rejects_an_empty_token_span(self):
        with pytest.raises(ValidationError, match="token_end"):
            Chunk.model_validate(self.valid(token_end=0))

    def test_rejects_text_that_disagrees_with_its_span(self):
        # The check that catches a chunk whose text came from anywhere other
        # than the span it claims -- the exact bug that would make every
        # relevance label silently resolve to the wrong chunk.
        with pytest.raises(ValidationError, match="does not match"):
            Chunk.model_validate(self.valid(text="a much longer string"))

    def test_is_frozen(self):
        chunk = Chunk.model_validate(self.valid())
        with pytest.raises(ValidationError):
            chunk.text = "mutated"
