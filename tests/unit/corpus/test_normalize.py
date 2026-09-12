"""Text normalization.

This is the highest-stakes "boring" module in the project. Golden-set relevance
labels are character offsets into normalized text, so any change to these rules
silently moves every label. That is why NORMALIZER_VERSION exists and why it is
part of the corpus manifest hash.

Idempotence is the property that matters most: normalize(normalize(x)) must
equal normalize(x), or re-ingesting a snapshot would drift offsets.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from groundtruth.corpus.normalize import NORMALIZER_VERSION, normalize_text


class TestUnicode:
    def test_applies_nfkc_compatibility_folding(self):
        # The ligature is a single codepoint; NFKC expands it. Without this,
        # a query for "affirmed" would miss a passage containing "aﬃrmed".
        assert normalize_text("aﬃrmed") == "affirmed"

    def test_folds_fullwidth_characters(self):
        # Written as escapes: fullwidth forms are visually identical to their
        # ASCII lookalikes, so a literal here would read as a no-op test.
        assert normalize_text("\uff28\uff25\uff2c\uff2c\uff2f") == "HELLO"

    def test_converts_non_breaking_space_to_a_plain_space(self):
        # NFKC already does this, but it is asserted explicitly because a
        # stray NBSP is a classic cause of a failed exact-quote match.
        # Written as an escape on purpose: a literal NBSP in source is
        # invisible, and this assertion would look like it tested nothing.
        assert normalize_text("v.\u00a0Smith") == "v. Smith"


class TestLineEndings:
    def test_converts_crlf_to_lf(self):
        assert normalize_text("a\r\nb") == "a\nb"

    def test_converts_lone_cr_to_lf(self):
        assert normalize_text("a\rb") == "a\nb"

    def test_mixed_line_endings_normalize_consistently(self):
        assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


class TestWhitespace:
    def test_collapses_runs_of_spaces(self):
        assert normalize_text("a     b") == "a b"

    def test_collapses_tabs_into_a_single_space(self):
        assert normalize_text("a\t\t\tb") == "a b"

    def test_strips_trailing_whitespace_on_each_line(self):
        assert normalize_text("a   \nb   ") == "a\nb"

    def test_preserves_a_single_paragraph_break(self):
        # Paragraph structure is meaningful in an opinion; collapsing it would
        # merge a holding into the preceding recitation of facts.
        assert normalize_text("a\n\nb") == "a\n\nb"

    def test_collapses_three_or_more_newlines_to_a_paragraph_break(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_text("\n\n  text  \n\n") == "text"

    def test_empty_input_yields_empty_output(self):
        assert normalize_text("") == ""

    def test_whitespace_only_input_yields_empty_output(self):
        assert normalize_text("  \n\t\n  ") == ""


class TestReporterArtifacts:
    def test_removes_bracketed_star_pagination(self):
        # Reporter page markers are layout, not language. Left in, they become
        # tokens that dilute the embedding and pollute BM25 term statistics.
        assert normalize_text("the court held [*1145] that summary") == (
            "the court held that summary"
        )

    def test_removes_bare_star_pagination_at_a_line_start(self):
        assert normalize_text("*1145\nThe court held") == "The court held"

    def test_does_not_remove_a_star_that_is_not_pagination(self):
        # A footnote marker or emphasis must survive: over-aggressive stripping
        # silently deletes content, which is far worse than leaving noise.
        assert normalize_text("see note *supra") == "see note *supra"

    def test_does_not_remove_a_number_that_is_not_pagination(self):
        assert normalize_text("28 U.S.C. 1331") == "28 U.S.C. 1331"


class TestIdempotence:
    def test_normalizing_twice_changes_nothing(self):
        raw = "  a\r\n\r\n\r\n\tb   [*12]  c\u00a0d  "
        once = normalize_text(raw)
        assert normalize_text(once) == once

    @given(st.text(max_size=400))
    def test_is_idempotent_for_arbitrary_text(self, raw: str):
        # The property that protects every golden-set character offset.
        once = normalize_text(raw)
        assert normalize_text(once) == once

    @given(st.text(max_size=400))
    def test_output_never_contains_carriage_returns_or_tabs(self, raw: str):
        out = normalize_text(raw)
        assert "\r" not in out
        assert "\t" not in out

    @given(st.text(max_size=400))
    def test_output_is_stripped(self, raw: str):
        out = normalize_text(raw)
        assert out == out.strip()


class TestVersioning:
    def test_version_is_declared(self):
        # Part of the corpus manifest hash. Changing the rules above without
        # bumping this would move every label with nothing to detect it.
        assert NORMALIZER_VERSION
        assert isinstance(NORMALIZER_VERSION, str)
