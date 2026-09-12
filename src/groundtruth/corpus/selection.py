"""Corpus selection and record mapping.

Deliberately separated from the HTTP client in ``courtlistener.py``. Everything
here is pure -- selection to query parameters, API record to Document -- so it
is fully tested, while the module that actually talks to the network stays
thin enough to be honestly excluded from coverage.

**Selection is deterministic by design.** A fixed court set, a fixed date
window, ordered by opinion id, taking the first N after fixed filters. Anything
resembling a random sample would make the corpus irreproducible, and the corpus
hash is an anti-cheat control (ADR-0010) -- it has to mean something.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field

from groundtruth.corpus.html_text import html_to_text, looks_like_html
from groundtruth.corpus.models import Document
from groundtruth.corpus.normalize import normalize_text
from groundtruth.frozen import FrozenModel

#: Sparse fieldset. The full record carries ~30 fields including several
#: alternate HTML renderings; requesting only what is used cuts the payload by
#: roughly an order of magnitude.
OPINION_FIELDS: Final[str] = ",".join(
    ["id", "cluster_id", "plain_text", "html_with_citations", "extracted_by_ocr", "absolute_url"]
)
CLUSTER_FIELDS: Final[str] = ",".join(["id", "case_name", "date_filed"])

PAGE_SIZE: Final[int] = 100

_SITE_ROOT: Final[str] = "https://www.courtlistener.com"


class CorpusSelection(FrozenModel):
    """A reproducible description of which opinions to fetch.

    Recorded alongside the corpus manifest so the selection that produced a
    snapshot is auditable rather than folklore.
    """

    courts: tuple[str, ...] = ("scotus",)
    date_filed_after: str = "1990-01-01"
    date_filed_before: str = "2020-12-31"
    precedential_status: str = "Published"
    limit: int = Field(default=500, gt=0)

    #: Stubs -- procedural orders, one-line dispositions -- carry no holding to
    #: retrieve and would make retrieval look artificially easy.
    min_chars: int = Field(default=2_000, ge=0)
    #: A handful of opinions run to hundreds of pages and would dominate the
    #: chunk population on their own.
    max_chars: int = Field(default=200_000, gt=0)
    #: OCR'd text is riddled with recognition errors. Including it would
    #: measure the OCR pipeline rather than the retrieval configuration.
    exclude_ocr: bool = True

    @property
    def corpus_id(self) -> str:
        courts = "+".join(self.courts)
        return (
            f"courtlistener-{courts}-"
            f"{self.date_filed_after}..{self.date_filed_before}-n{self.limit}"
        )


def opinion_query(selection: CorpusSelection) -> dict[str, str]:
    """Build the opinion list query."""
    return {
        "cluster__docket__court": ",".join(selection.courts),
        "cluster__date_filed__gte": selection.date_filed_after,
        "cluster__date_filed__lte": selection.date_filed_before,
        "cluster__precedential_status": selection.precedential_status,
        # Ordering by id is what makes "the first N" a stable set rather than
        # whatever the server happened to return.
        "order_by": "id",
        "page_size": str(PAGE_SIZE),
        "fields": OPINION_FIELDS,
    }


def cluster_query(selection: CorpusSelection) -> dict[str, str]:
    """Build the cluster list query used to collect case names and dates.

    Metadata is gathered from the paginated LIST endpoint rather than one
    request per cluster. The API rejects ``id__in``, so per-cluster fetching
    would cost one request per document -- and CourtListener's free tier
    allows only 50 requests per hour, which makes that approach take hours for
    a corpus this size. Filtering the list by the same court and date window
    costs a couple of requests instead.

    Note the filter names differ from the opinion query: on ``/clusters/`` the
    fields are top level (``docket__court``), whereas ``/opinions/`` reaches
    them through its cluster (``cluster__docket__court``).
    """
    return {
        "docket__court": ",".join(selection.courts),
        "date_filed__gte": selection.date_filed_after,
        "date_filed__lte": selection.date_filed_before,
        "precedential_status": selection.precedential_status,
        "order_by": "id",
        "page_size": str(PAGE_SIZE),
        "fields": CLUSTER_FIELDS,
    }


def extract_body(record: Mapping[str, Any]) -> str:
    """Get an opinion's normalized text, preferring plain text over markup.

    When ``plain_text`` is empty the HTML rendering is reduced rather than
    skipping the record, because dropping every HTML-only opinion would bias
    the corpus toward whichever courts happen to supply plain text.
    """
    plain = str(record.get("plain_text") or "").strip()
    if plain:
        return normalize_text(plain)

    markup = str(record.get("html_with_citations") or "").strip()
    if markup:
        return normalize_text(html_to_text(markup) if looks_like_html(markup) else markup)

    return ""


def is_eligible(record: Mapping[str, Any], selection: CorpusSelection, body: str) -> bool:
    """Whether a fetched record belongs in the corpus."""
    if selection.exclude_ocr and record.get("extracted_by_ocr"):
        return False
    return selection.min_chars <= len(body) <= selection.max_chars


def to_document(
    record: Mapping[str, Any],
    *,
    court: str,
    clusters: Mapping[int, Mapping[str, Any]] | None = None,
) -> Document | None:
    """Map an API record to a Document, or None if it carries no usable text."""
    text = extract_body(record)
    if not text:
        return None

    cluster_id = int(record.get("cluster_id") or 0)
    cluster = (clusters or {}).get(cluster_id, {})
    url = str(record.get("absolute_url") or "")

    return Document(
        doc_id=f"cl-{record['id']}",
        text=text,
        title=str(cluster.get("case_name") or ""),
        court=court,
        date_filed=str(cluster.get("date_filed") or ""),
        source_url=f"{_SITE_ROOT}{url}" if url else "",
    )
