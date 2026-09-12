"""HTML-to-text extraction for opinion bodies."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from groundtruth.corpus.html_text import html_to_text, looks_like_html
from groundtruth.corpus.normalize import normalize_text


class TestDetection:
    def test_detects_markup(self):
        assert looks_like_html("<p>The court held</p>")

    def test_detects_a_bare_break(self):
        assert looks_like_html("line one<br>line two")

    def test_plain_text_is_not_markup(self):
        assert not looks_like_html("The court held that summary judgment was proper.")

    def test_a_less_than_sign_is_not_markup(self):
        # Opinions contain arithmetic and citations with angle brackets.
        # Treating "damages < $500" as HTML would strip real content.
        assert not looks_like_html("damages < $500 are de minimis")


class TestExtraction:
    def test_strips_tags_and_keeps_text(self):
        assert html_to_text("<p>The court held</p>").strip() == "The court held"

    def test_preserves_paragraph_boundaries(self):
        # The holding must not run into the facts that precede it.
        out = html_to_text("<p>Facts here.</p><p>We therefore reverse.</p>")
        assert "Facts here." in out
        assert "We therefore reverse." in out
        assert "\n" in out.strip()

    def test_treats_a_line_break_as_a_boundary(self):
        assert "\n" in html_to_text("one<br/>two")

    def test_drops_script_and_style_content(self):
        # Their content is machinery, not opinion text. Left in, it becomes
        # tokens that pollute BM25 term statistics.
        out = html_to_text("<p>Held.</p><script>var x = 1;</script><style>p{color:red}</style>")
        assert "var x" not in out
        assert "color:red" not in out
        assert "Held." in out

    def test_drops_comments(self):
        assert "hidden" not in html_to_text("<p>Held.</p><!-- hidden note -->")

    def test_unescapes_entities(self):
        assert html_to_text("<p>Smith &amp; Jones</p>").strip() == "Smith & Jones"

    def test_unescapes_numeric_entities(self):
        assert "§" in html_to_text("<p>&#167; 1983</p>")

    def test_entity_that_decodes_to_a_tag_is_not_re_stripped(self):
        # Unescaping before stripping would let "&lt;b&gt;" become a real tag
        # and silently delete the word between them.
        assert "<b>emphasis</b>" in html_to_text("<p>&lt;b&gt;emphasis&lt;/b&gt;</p>")

    def test_empty_input_yields_empty_output(self):
        assert html_to_text("") == ""

    def test_plain_text_passes_through_unchanged(self):
        assert html_to_text("No markup at all.").strip() == "No markup at all."

    def test_nested_markup_keeps_all_text(self):
        out = html_to_text("<div><p>The <em>court</em> <b>held</b>.</p></div>")
        assert "court" in out
        assert "held" in out


class TestCompositionWithNormalize:
    """Extraction feeds normalization; together they must be well behaved."""

    def test_extraction_then_normalization_is_clean(self):
        markup = "<div><p>The  court   held.</p>\n\n\n<p>We reverse.</p></div>"
        out = normalize_text(html_to_text(markup))
        assert out == "The court held.\n\nWe reverse."

    @given(st.text(max_size=300))
    def test_never_raises_on_arbitrary_input(self, raw: str):
        # Corpus fetch must not die on one malformed record out of 500.
        normalize_text(html_to_text(raw))
