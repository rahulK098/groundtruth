"""Passage sampling, and turning a model proposal into a candidate.

No API key and no network: everything the Anthropic call does *around* these
functions is plumbing, and everything that can be wrong is here.
"""

from __future__ import annotations

import pytest

from groundtruth.corpus.models import Document
from groundtruth.golden.proposals import (
    MAX_QUOTE_CHARS,
    RejectedProposal,
    candidate_from_proposal,
)
from groundtruth.golden.sampling import (
    DEFAULT_WINDOW_CHARS,
    SampledPassage,
    SamplingError,
    sample_passages,
)

HOLDING = "Summary judgment is appropriate only where no genuine dispute exists."
PASSAGE_TEXT = f"Before the analysis. {HOLDING} After the analysis, at some length."
PASSAGE = SampledPassage(
    doc_id="cl-1", char_start=5000, char_end=5000 + len(PASSAGE_TEXT), text=PASSAGE_TEXT
)


def documents(count: int = 5, length: int = 8000) -> tuple[Document, ...]:
    return tuple(
        Document(doc_id=f"cl-{i:03d}", text=f"opinion {i} " * (length // 10)) for i in range(count)
    )


def proposal(**kwargs: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "query": "when can a case be decided without a trial",
        "category": "factual-lookup",
        "spans": [{"quote": HOLDING, "gain": 3}],
    }
    raw.update(kwargs)
    return raw


def make(raw: dict[str, object], passage: SampledPassage = PASSAGE):
    return candidate_from_proposal(
        raw,
        passage,
        candidate_id="c-001",
        generator_model="claude-sonnet-5-20260101",
        prompt_version="golden-candidate-v1",
    )


class TestSampling:
    def test_is_reproducible_for_one_seed(self):
        docs = documents()
        assert sample_passages(docs, count=10) == sample_passages(docs, count=10)

    def test_a_different_seed_gives_different_passages(self):
        docs = documents()
        assert sample_passages(docs, count=10, seed=1) != sample_passages(docs, count=10, seed=2)

    def test_does_not_depend_on_document_order(self):
        docs = documents()
        assert sample_passages(docs, count=10) == sample_passages(tuple(reversed(docs)), count=10)

    def test_spreads_across_documents_before_repeating_any(self):
        # Opinion lengths vary by an order of magnitude; uniform sampling over
        # the corpus would concentrate the set in a few long cases.
        passages = sample_passages(documents(5), count=5)
        assert len({p.doc_id for p in passages}) == 5

    def test_window_is_not_aligned_to_any_evaluated_chunking(self):
        # 1500 characters is roughly 375 tokens: neither the 512- nor the
        # 256-token configuration, so quoted spans do not tend to sit inside
        # one configuration's boundaries.
        assert DEFAULT_WINDOW_CHARS == 1500
        assert all(p.char_end - p.char_start == 1500 for p in sample_passages(documents(), count=3))

    def test_skips_documents_too_short_to_carry_a_holding(self):
        docs = (*documents(2), Document(doc_id="cl-stub", text="short"))
        assert "cl-stub" not in {p.doc_id for p in sample_passages(docs, count=4)}

    def test_refuses_a_corpus_with_nothing_substantial(self):
        with pytest.raises(SamplingError, match="nothing substantial"):
            sample_passages((Document(doc_id="cl-1", text="tiny"),), count=1)

    def test_rejects_a_non_positive_count(self):
        with pytest.raises(SamplingError, match="count"):
            sample_passages(documents(), count=0)

    def test_passages_carry_the_text_at_the_offsets_they_claim(self):
        docs = documents()
        by_id = {d.doc_id: d for d in docs}
        for passage in sample_passages(docs, count=6):
            source = by_id[passage.doc_id].text
            assert passage.text == source[passage.char_start : passage.char_end]


class TestCandidateFromProposal:
    def test_resolves_a_quote_to_absolute_document_offsets(self):
        # The model sees a window; labels must point into the whole document.
        candidate = make(proposal())
        assert not isinstance(candidate, RejectedProposal)
        label = candidate.labels[0]
        assert label.char_start == 5000 + PASSAGE_TEXT.index(HOLDING)
        assert label.quote == HOLDING

    def test_the_resolved_span_round_trips_through_the_passage(self):
        candidate = make(proposal())
        assert not isinstance(candidate, RejectedProposal)
        label = candidate.labels[0]
        offset = label.char_start - PASSAGE.char_start
        assert PASSAGE.text[offset : offset + len(label.quote)] == label.quote

    def test_records_the_generator_and_the_source_window(self):
        candidate = make(proposal())
        assert not isinstance(candidate, RejectedProposal)
        assert candidate.prompt_version == "golden-candidate-v1"
        assert (candidate.source_char_start, candidate.source_char_end) == (
            PASSAGE.char_start,
            PASSAGE.char_end,
        )

    def test_accepts_several_graded_spans(self):
        candidate = make(
            proposal(
                spans=[
                    {"quote": HOLDING, "gain": 3},
                    {"quote": "After the analysis, at some length.", "gain": 1},
                ]
            )
        )
        assert not isinstance(candidate, RejectedProposal)
        assert [lbl.gain for lbl in candidate.labels] == [3, 1]


class TestRejection:
    def test_a_declined_passage_is_recorded_not_dropped(self):
        # The rejection rate is evidence (ADR-0009); a filter that discarded
        # silently would make "generated 150, accepted 62" a lie by omission.
        rejected = make({"decline": True, "decline_reason": "syllabus boilerplate"})
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "declined"
        assert "boilerplate" in rejected.detail

    def test_a_quote_that_is_not_verbatim_is_rejected(self):
        # Resolving to a best guess would put a label at the wrong offsets,
        # which nothing downstream can detect.
        rejected = make(proposal(spans=[{"quote": "Summary judgment is always fine.", "gain": 3}]))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "quote-not-found"

    def test_a_quote_appearing_twice_is_rejected_as_ambiguous(self):
        repeated = "The same sentence appears twice here. "
        passage = SampledPassage(
            doc_id="cl-1", char_start=0, char_end=len(repeated) * 2, text=repeated * 2
        )
        rejected = make(proposal(spans=[{"quote": repeated.strip(), "gain": 3}]), passage)
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "quote-ambiguous"

    def test_an_oversized_quote_is_rejected(self):
        # A span that size is the passage. Any chunk overlapping it satisfies
        # span-level Recall, which would lift every configuration at once.
        long_text = "x" * (MAX_QUOTE_CHARS + 50)
        passage = SampledPassage(
            doc_id="cl-1", char_start=0, char_end=len(long_text), text=long_text
        )
        rejected = make(proposal(spans=[{"quote": long_text, "gain": 3}]), passage)
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "quote-too-long"

    def test_a_fragment_quote_is_rejected(self):
        rejected = make(proposal(spans=[{"quote": "judgment", "gain": 3}]))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "quote-too-short"

    def test_an_empty_query_is_rejected(self):
        rejected = make(proposal(query="   "))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "no-query"

    def test_an_unknown_category_is_rejected(self):
        rejected = make(proposal(category="interesting"))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "bad-category"

    @pytest.mark.parametrize("spans", [[], None, "not a list"])
    def test_a_proposal_without_spans_is_rejected(self, spans: object):
        rejected = make(proposal(spans=spans))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "no-spans"

    @pytest.mark.parametrize("gain", [0, 4, "three", None])
    def test_a_gain_outside_the_rubric_is_rejected(self, gain: object):
        # 0 too: a proposal whose only span is "not relevant" is not a
        # proposal, and the tool schema asks for 1-3.
        rejected = make(proposal(spans=[{"quote": HOLDING, "gain": gain}]))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "bad-gain"

    def test_two_labels_on_one_span_are_rejected(self):
        rejected = make(
            proposal(spans=[{"quote": HOLDING, "gain": 3}, {"quote": HOLDING, "gain": 2}])
        )
        assert isinstance(rejected, RejectedProposal)
        assert rejected.code == "duplicate-span"

    def test_a_rejection_names_the_passage_it_came_from(self):
        rejected = make(proposal(query=""))
        assert isinstance(rejected, RejectedProposal)
        assert rejected.passage_id == PASSAGE.passage_id
