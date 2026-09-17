"""Golden-set value objects.

These validators are the highest-leverage code in the project. The golden set
is what every metric is computed against: if a label is wrong, the comparison
table, the gate and the final recommendation are all wrong in ways no
downstream rigor can detect.

So they reject *incoherent* records, not merely malformed ones -- a pair whose
provenance claims it was accepted unchanged while listing edited fields is
syntactically fine and semantically a lie.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundtruth.golden.models import (
    CATEGORIES,
    MIN_RELEVANT_GAIN,
    GoldenPair,
    GoldenSet,
    Provenance,
    RelevanceLabel,
)

QUOTE = "Summary judgment is appropriate"


def label(
    doc_id: str = "cl-1", start: int = 100, quote: str = QUOTE, gain: int = 3
) -> RelevanceLabel:
    return RelevanceLabel(
        doc_id=doc_id, char_start=start, char_end=start + len(quote), gain=gain, quote=quote
    )


def provenance(**kwargs: object) -> Provenance:
    fields: dict[str, object] = {
        "origin": "human",
        "reviewer": "rahul",
        "reviewed_at": "2026-09-13T10:00:00+00:00",
        "review_action": "accepted",
    }
    fields.update(kwargs)
    return Provenance(**fields)  # type: ignore[arg-type]


def pair(query_id: str = "q-001", query: str | None = None, **kwargs) -> GoldenPair:
    # The query text defaults off the id: the set rejects duplicate text, so a
    # shared default would make every multi-pair fixture illegal.
    fields: dict[str, object] = {
        "query_id": query_id,
        "query": query if query is not None else f"summary judgment standard {query_id}",
        "category": "factual-lookup",
        "labels": (label(),),
        "provenance": provenance(),
    }
    fields.update(kwargs)
    return GoldenPair(**fields)  # type: ignore[arg-type]


class TestRelevanceLabel:
    def test_quote_length_must_match_the_span(self):
        # The single most valuable integrity check in the project: it catches
        # corpus re-normalization drift immediately instead of as an
        # unexplained metric shift weeks later.
        with pytest.raises(ValidationError, match="quote"):
            RelevanceLabel(doc_id="cl-1", char_start=0, char_end=5, gain=3, quote="much longer")

    def test_rejects_an_inverted_span(self):
        with pytest.raises(ValidationError, match="char_end"):
            RelevanceLabel(doc_id="cl-1", char_start=50, char_end=10, gain=3, quote="")

    @pytest.mark.parametrize("gain", [-1, 4])
    def test_rejects_a_gain_outside_the_rubric(self, gain: int):
        with pytest.raises(ValidationError):
            label(gain=gain)

    def test_a_zero_gain_label_is_valid(self):
        # Explicit "not relevant" is a real judgment worth recording: it is
        # the difference between "reviewed and rejected" and "never looked at".
        assert label(gain=0).gain == 0

    def test_knows_whether_it_counts_as_relevant(self):
        # Recall binarizes here; nDCG uses the full grade (ADR-0004).
        assert label(gain=MIN_RELEVANT_GAIN).is_relevant
        assert not label(gain=MIN_RELEVANT_GAIN - 1).is_relevant


class TestGoldenPair:
    def test_accepts_a_well_formed_pair(self):
        assert pair().query_id == "q-001"

    def test_requires_at_least_one_relevant_span(self):
        # A query with nothing at gain >= 2 is unscoreable by Recall: it would
        # be excluded from the mean, so it contributes nothing but a footnote.
        with pytest.raises(ValidationError, match="gain"):
            pair(labels=(label(gain=1),))

    def test_requires_at_least_one_label(self):
        with pytest.raises(ValidationError):
            pair(labels=())

    def test_rejects_two_labels_on_the_same_span(self):
        with pytest.raises(ValidationError, match="duplicate"):
            pair(labels=(label(), label()))

    def test_allows_two_labels_on_different_spans_of_one_document(self):
        # Multi-hop queries need exactly this.
        assert len(pair(labels=(label(start=100), label(start=900))).labels) == 2

    def test_orders_labels_canonically(self):
        # Hashing is over the serialized pair, so label order has to be a
        # function of the labels rather than of the order they were entered.
        out = pair(labels=(label(doc_id="cl-9", start=10), label(doc_id="cl-2", start=500)))
        assert [(lbl.doc_id, lbl.char_start) for lbl in out.labels] == [
            ("cl-2", 500),
            ("cl-9", 10),
        ]

    def test_rejects_an_unknown_category(self):
        with pytest.raises(ValidationError):
            pair(category="vibes")

    def test_rejects_a_blank_query(self):
        with pytest.raises(ValidationError):
            pair(query="   ")

    def test_every_declared_category_is_usable(self):
        for category in CATEGORIES:
            assert pair(category=category).category == category


class TestProvenance:
    def test_an_edit_must_say_what_was_edited(self):
        # "edited" with nothing listed is indistinguishable from "accepted",
        # and the report breaks results down by exactly that distinction.
        with pytest.raises(ValidationError, match="edited_fields"):
            provenance(review_action="edited")

    def test_an_acceptance_must_not_claim_edits(self):
        with pytest.raises(ValidationError, match="edited_fields"):
            provenance(review_action="accepted", edited_fields=("query",))

    def test_llm_origin_must_name_its_generator_and_prompt(self):
        # Without both, "generated by a model" is unfalsifiable and the
        # human-vs-LLM split in the report means nothing.
        with pytest.raises(ValidationError, match="generator_model"):
            provenance(origin="llm")
        with pytest.raises(ValidationError, match="prompt_version"):
            provenance(origin="llm", generator_model="claude-x", source_candidate_id="c-1")

    def test_human_origin_must_not_claim_a_generator(self):
        with pytest.raises(ValidationError, match="generator_model"):
            provenance(origin="human", generator_model="claude-x")

    def test_requires_a_reviewer(self):
        # ADR-0009: every accepted pair carries who accepted it.
        with pytest.raises(ValidationError):
            provenance(reviewer="")

    def test_a_reviewed_llm_pair_is_valid(self):
        prov = provenance(
            origin="llm",
            generator_model="claude-sonnet-5-20260101",
            prompt_version="golden-candidate-v1",
            source_candidate_id="c-042",
            review_action="edited",
            edited_fields=("query", "labels"),
        )
        assert prov.origin == "llm"


class TestGoldenSet:
    def test_rejects_a_duplicate_query_id(self):
        with pytest.raises(ValidationError, match="query_id"):
            GoldenSet(pairs=(pair("q-1"), pair("q-1", query="different text")))

    def test_rejects_a_duplicate_query_regardless_of_case_and_spacing(self):
        # Two spellings of one question are one data point reported as two,
        # and they inflate whichever category they land in.
        with pytest.raises(ValidationError, match="duplicate query"):
            GoldenSet(
                pairs=(
                    pair("q-1", query="Summary judgment standard"),
                    pair("q-2", query="  summary   judgment standard "),
                )
            )

    def test_orders_pairs_by_query_id(self):
        assert [p.query_id for p in GoldenSet(pairs=(pair("q-2"), pair("q-1"))).pairs] == [
            "q-1",
            "q-2",
        ]

    def test_an_empty_set_is_representable(self):
        # Phase 5 starts here, and a type that cannot represent "not built yet"
        # forces a placeholder that later gets mistaken for data.
        assert GoldenSet().pairs == ()

    def test_hash_is_stable_across_construction_order(self):
        left = GoldenSet(pairs=(pair("q-1"), pair("q-2")))
        right = GoldenSet(pairs=(pair("q-2"), pair("q-1")))
        assert left.golden_set_hash == right.golden_set_hash

    def test_hash_changes_when_a_label_changes(self):
        # This is the anti-cheat the gate relies on: editing the golden set to
        # dodge a regression must be detectable (ADR-0010).
        base = GoldenSet(pairs=(pair("q-1"),))
        edited = GoldenSet(pairs=(pair("q-1", labels=(label(gain=2),)),))
        assert base.golden_set_hash != edited.golden_set_hash

    def test_hash_ignores_review_timestamps(self):
        # Re-reviewing an unchanged pair must not invalidate every committed
        # baseline. What the set *is* does not include when it was looked at.
        base = GoldenSet(pairs=(pair("q-1"),))
        later = GoldenSet(
            pairs=(pair("q-1", provenance=provenance(reviewed_at="2027-01-01T00:00:00+00:00")),)
        )
        assert base.golden_set_hash == later.golden_set_hash

    def test_hash_is_prefixed_with_its_algorithm(self):
        assert GoldenSet(pairs=(pair(),)).golden_set_hash.startswith("sha256:")

    def test_counts_by_category(self):
        counts = GoldenSet(
            pairs=(
                pair("q-1", category="factual-lookup"),
                pair("q-2", query="b", category="factual-lookup"),
                pair("q-3", query="c", category="multi-hop"),
            )
        ).counts_by_category()
        assert counts["factual-lookup"] == 2
        assert counts["multi-hop"] == 1
        # Every category is present, including the empty ones -- otherwise a
        # missing category reads as zero only if you already knew to look.
        assert set(counts) == set(CATEGORIES)

    def test_counts_by_origin(self):
        counts = GoldenSet(
            pairs=(
                pair("q-1"),
                pair(
                    "q-2",
                    query="b",
                    provenance=provenance(
                        origin="llm",
                        generator_model="claude-x",
                        prompt_version="v1",
                    ),
                ),
            )
        ).counts_by_origin()
        assert counts == {"human": 1, "llm": 1}
