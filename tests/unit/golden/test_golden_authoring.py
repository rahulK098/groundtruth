"""Hand-authored pairs.

The ~30 attorney-phrased queries never pass through the generator, so they
need their own path into the review log. The properties that matter: the pair
is marked ``origin="human"`` with no generator fields, every quote is resolved
against the *whole* document rather than a sampled window, and a quote that
cannot be located is refused rather than guessed at.
"""

from __future__ import annotations

import pytest

from groundtruth.golden.authoring import (
    AuthoringError,
    from_authored,
    next_human_query_id,
    to_template,
)
from groundtruth.golden.candidates import ReviewDecision
from groundtruth.golden.review import ReviewContext
from tests.fixtures.mini_corpus import mini_documents

CONTEXT = ReviewContext(reviewer="rahul", decided_at="2026-09-17T10:00:00+00:00")

QUOTE = (
    "Summary judgment is appropriate only where there is no\n"
    "genuine dispute as to any material fact"
)


def documents() -> dict[str, object]:
    return {doc.doc_id: doc for doc in mini_documents()}


def authored(
    query: str = "what standard applies on a motion for summary judgment",
    category: str = "factual-lookup",
    spans: list[dict[str, object]] | None = None,
) -> str:
    import yaml

    body = {
        "query": query,
        "category": category,
        "spans": spans
        if spans is not None
        else [{"doc_id": "mini-001", "gain": 3, "quote": QUOTE}],
    }
    return yaml.safe_dump(body, sort_keys=False)


class TestTemplate:
    def test_template_parses_as_yaml_with_the_expected_keys(self):
        import yaml

        raw = yaml.safe_load(to_template())
        assert set(raw) == {"query", "category", "spans"}
        assert raw["spans"][0].keys() == {"doc_id", "gain", "quote"}

    def test_template_names_every_category(self):
        text = to_template()
        for category in (
            "factual-lookup",
            "multi-hop",
            "negation-exclusion",
            "procedural",
            "ambiguous-terminology",
        ):
            assert category in text


class TestFromAuthored:
    def test_produces_a_human_origin_decision_with_no_candidate(self):
        decision = from_authored(authored(), documents(), query_id="q-h-001", context=CONTEXT)

        assert isinstance(decision, ReviewDecision)
        assert decision.candidate_id is None
        assert decision.action == "accept"
        assert decision.subject == "query:q-h-001"

        pair = decision.pair
        assert pair is not None
        assert pair.provenance.origin == "human"
        assert pair.provenance.generator_model is None
        assert pair.provenance.prompt_version is None
        assert pair.provenance.reviewer == "rahul"
        assert pair.provenance.review_action == "accepted"

    def test_quotes_resolve_to_offsets_in_the_full_document(self):
        decision = from_authored(authored(), documents(), query_id="q-h-001", context=CONTEXT)
        label = decision.pair.labels[0]  # type: ignore[union-attr]

        text = documents()["mini-001"].text  # type: ignore[attr-defined]
        assert text[label.char_start : label.char_end] == QUOTE
        assert label.doc_id == "mini-001"
        assert label.gain == 3

    def test_spans_may_point_at_different_documents(self):
        second = "Purposeful availment is the touchstone."
        spans = [
            {"doc_id": "mini-001", "gain": 3, "quote": QUOTE},
            {"doc_id": "mini-002", "gain": 2, "quote": second},
        ]
        decision = from_authored(
            authored(category="multi-hop", spans=spans),
            documents(),
            query_id="q-h-002",
            context=CONTEXT,
        )
        assert {lbl.doc_id for lbl in decision.pair.labels} == {"mini-001", "mini-002"}  # type: ignore[union-attr]

    def test_unknown_document_is_refused(self):
        spans = [{"doc_id": "mini-999", "gain": 3, "quote": QUOTE}]
        with pytest.raises(AuthoringError, match="mini-999"):
            from_authored(authored(spans=spans), documents(), query_id="q-h-001", context=CONTEXT)

    def test_quote_not_in_document_is_refused_not_guessed(self):
        spans = [{"doc_id": "mini-001", "gain": 3, "quote": "this sentence appears nowhere at all"}]
        with pytest.raises(AuthoringError, match="quote-not-found"):
            from_authored(authored(spans=spans), documents(), query_id="q-h-001", context=CONTEXT)

    def test_ambiguous_quote_is_refused(self):
        # Two sentences in mini-001 end "...the nonmoving\nparty." -- long
        # enough to clear the minimum quote length, so the ambiguity check is
        # the one that fires.
        repeated = "the nonmoving\nparty."
        text = documents()["mini-001"].text  # type: ignore[attr-defined]
        assert text.count(repeated) == 2

        spans = [{"doc_id": "mini-001", "gain": 3, "quote": repeated}]
        with pytest.raises(AuthoringError, match="quote-ambiguous"):
            from_authored(authored(spans=spans), documents(), query_id="q-h-001", context=CONTEXT)

    def test_pair_must_carry_a_relevant_span(self):
        spans = [{"doc_id": "mini-001", "gain": 1, "quote": QUOTE}]
        with pytest.raises(AuthoringError, match="gain"):
            from_authored(authored(spans=spans), documents(), query_id="q-h-001", context=CONTEXT)

    @pytest.mark.parametrize(
        "text, fragment",
        [
            ("not: [valid yaml", "YAML"),
            ("- just\n- a list\n", "mapping"),
            (authored(query="   "), "query"),
            (authored(category="made-up"), "category"),
            (authored(spans=[]), "spans"),
        ],
    )
    def test_malformed_input_names_the_problem(self, text: str, fragment: str):
        with pytest.raises(AuthoringError, match=fragment):
            from_authored(text, documents(), query_id="q-h-001", context=CONTEXT)


class TestNextHumanQueryId:
    def test_starts_at_one_with_an_empty_log(self):
        assert next_human_query_id(()) == "q-h-001"

    def test_counts_only_human_authored_subjects(self):
        first = from_authored(authored(), documents(), query_id="q-h-001", context=CONTEXT)
        assert next_human_query_id((first,)) == "q-h-002"

    def test_never_reuses_an_id_already_in_the_log(self):
        # A gap in the sequence must not be filled: replaying the log with a
        # reused id would collapse two distinct pairs into one subject.
        seventh = from_authored(authored(), documents(), query_id="q-h-007", context=CONTEXT)
        assert next_human_query_id((seventh,)) == "q-h-008"
