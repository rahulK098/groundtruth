"""Corpus selection and API record mapping.

Pure logic, deliberately separated from the HTTP client so it can be tested
properly. Selection determinism matters more than it looks: the corpus hash is
an anti-cheat control, so "which documents are in the corpus" must be a
function of the selection, never of when the fetch happened to run.
"""

from __future__ import annotations

import pytest

from groundtruth.corpus.selection import (
    CorpusSelection,
    cluster_query,
    extract_body,
    is_eligible,
    opinion_query,
    to_document,
)


def record(**overrides):
    base = {
        "id": 42,
        "cluster_id": 7,
        "plain_text": "The court held that summary judgment was proper.",
        "html_with_citations": "",
        "extracted_by_ocr": False,
        "absolute_url": "/opinion/42/smith-v-jones/",
    }
    return {**base, **overrides}


class TestSelection:
    def test_corpus_id_encodes_the_whole_selection(self):
        # The id appears in the manifest; it has to distinguish two different
        # selections or the snapshot's provenance is a guess.
        sel = CorpusSelection(courts=("scotus",), limit=500)
        assert "scotus" in sel.corpus_id
        assert "n500" in sel.corpus_id
        assert "1990-01-01" in sel.corpus_id

    def test_corpus_id_differs_when_the_window_differs(self):
        a = CorpusSelection(date_filed_after="1990-01-01")
        b = CorpusSelection(date_filed_after="2000-01-01")
        assert a.corpus_id != b.corpus_id

    def test_is_frozen(self):
        from pydantic import ValidationError

        sel = CorpusSelection()
        with pytest.raises(ValidationError):
            sel.limit = 10

    def test_rejects_a_non_positive_limit(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CorpusSelection(limit=0)


class TestQuery:
    def test_orders_by_id(self):
        # Without a stable order, "the first N" is whatever the server felt
        # like returning, and the corpus stops being reproducible.
        assert opinion_query(CorpusSelection())["order_by"] == "id"

    def test_passes_the_date_window_through(self):
        q = opinion_query(CorpusSelection(date_filed_after="1995-01-01"))
        assert q["cluster__date_filed__gte"] == "1995-01-01"

    def test_joins_multiple_courts(self):
        q = opinion_query(CorpusSelection(courts=("scotus", "ca1")))
        assert q["cluster__docket__court"] == "scotus,ca1"

    def test_requests_a_sparse_fieldset(self):
        # The full record is ~30 fields with several alternate HTML renderings.
        q = opinion_query(CorpusSelection())
        assert "plain_text" in q["fields"]
        assert "xml_harvard" not in q["fields"]


class TestClusterQuery:
    """Metadata comes from the LIST endpoint, not one request per cluster.

    CourtListener's free tier allows 50 requests/hour. Per-cluster fetching
    would cost one request per document, so a 150-document corpus would take
    hours and look like a hung process.
    """

    def test_filters_are_top_level_not_cluster_prefixed(self):
        # On /clusters/ the fields are top level; on /opinions/ they are
        # reached through the cluster. Using the opinion spelling here returns
        # 400 "Unknown filter parameters are not allowed".
        q = cluster_query(CorpusSelection(courts=("scotus",)))
        assert q["docket__court"] == "scotus"
        assert "cluster__docket__court" not in q

    def test_uses_the_same_date_window_as_the_opinion_query(self):
        sel = CorpusSelection(date_filed_after="1995-01-01", date_filed_before="2005-12-31")
        q = cluster_query(sel)
        assert q["date_filed__gte"] == "1995-01-01"
        assert q["date_filed__lte"] == "2005-12-31"

    def test_orders_by_id(self):
        # The fetcher stops paging once past the highest cluster id it needs,
        # which is only sound if results are ordered by id.
        assert cluster_query(CorpusSelection())["order_by"] == "id"

    def test_requests_only_the_metadata_fields_used(self):
        q = cluster_query(CorpusSelection())
        assert q["fields"] == "id,case_name,date_filed"

    def test_pages_at_the_maximum_size(self):
        # Fewer, larger pages against a 50/hour budget.
        assert cluster_query(CorpusSelection())["page_size"] == "100"


class TestExtractBody:
    def test_prefers_plain_text(self):
        assert extract_body(record()).startswith("The court held")

    def test_falls_back_to_html(self):
        # Skipping HTML-only records would bias the corpus toward whichever
        # courts happen to supply plain text.
        body = extract_body(
            record(plain_text="", html_with_citations="<p>Held for the plaintiff.</p>")
        )
        assert body == "Held for the plaintiff."

    def test_normalizes_the_result(self):
        assert extract_body(record(plain_text="a   b\r\nc")) == "a b\nc"

    def test_returns_empty_when_there_is_no_body(self):
        assert extract_body(record(plain_text="", html_with_citations="")) == ""

    def test_treats_whitespace_only_text_as_empty(self):
        assert extract_body(record(plain_text="   \n  ", html_with_citations="")) == ""


class TestEligibility:
    def test_accepts_a_normal_opinion(self):
        sel = CorpusSelection(min_chars=10, max_chars=1000)
        assert is_eligible(record(), sel, "x" * 100)

    def test_rejects_a_stub(self):
        # One-line dispositions carry no holding and make retrieval look easy.
        sel = CorpusSelection(min_chars=2000)
        assert not is_eligible(record(), sel, "Affirmed.")

    def test_rejects_an_enormous_opinion(self):
        sel = CorpusSelection(max_chars=1000)
        assert not is_eligible(record(), sel, "x" * 5000)

    def test_rejects_ocr_text_by_default(self):
        # OCR errors would make the harness measure the OCR pipeline.
        sel = CorpusSelection(min_chars=0)
        assert not is_eligible(record(extracted_by_ocr=True), sel, "x" * 100)

    def test_accepts_ocr_when_explicitly_allowed(self):
        sel = CorpusSelection(min_chars=0, exclude_ocr=False)
        assert is_eligible(record(extracted_by_ocr=True), sel, "x" * 100)


class TestToDocument:
    def test_builds_a_prefixed_document_id(self):
        doc = to_document(record(), court="scotus")
        assert doc is not None
        assert doc.doc_id == "cl-42"

    def test_builds_an_absolute_source_url(self):
        doc = to_document(record(), court="scotus")
        assert doc is not None
        assert doc.source_url == "https://www.courtlistener.com/opinion/42/smith-v-jones/"

    def test_attaches_cluster_metadata(self):
        clusters = {7: {"case_name": "Smith v. Jones", "date_filed": "1999-05-04"}}
        doc = to_document(record(), court="scotus", clusters=clusters)
        assert doc is not None
        assert doc.title == "Smith v. Jones"
        assert doc.date_filed == "1999-05-04"

    def test_survives_missing_cluster_metadata(self):
        # Metadata is best effort; losing a case name must not lose the opinion.
        doc = to_document(record(), court="scotus", clusters={})
        assert doc is not None
        assert doc.title == ""
        assert doc.text

    def test_returns_none_when_there_is_no_text(self):
        assert to_document(record(plain_text="", html_with_citations=""), court="scotus") is None

    def test_text_is_normalized(self):
        doc = to_document(record(plain_text="a   b"), court="scotus")
        assert doc is not None
        assert doc.text == "a b"
