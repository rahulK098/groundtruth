"""The lexical analyzer.

Small surface, but it decides what BM25 can ever match, so its edges are
pinned rather than assumed.
"""

from __future__ import annotations

from groundtruth.index.analysis import tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Summary JUDGMENT") == ("summary", "judgment")

    def test_splits_on_punctuation_and_whitespace(self):
        assert tokenize("plaintiff's motion, denied.") == ("plaintiff", "s", "motion", "denied")

    def test_keeps_digits(self):
        # Legal text is full of them: section numbers, years, reporter cites.
        assert tokenize("Rule 56(c), 1994") == ("rule", "56", "c", "1994")

    def test_keeps_accented_letters_as_one_token(self):
        # A word-character class that dropped these would silently split
        # party names into fragments that no query could match.
        assert tokenize("Nuñez café") == ("nuñez", "café")

    def test_underscore_is_a_separator_not_a_letter(self):
        assert tokenize("doc_id") == ("doc", "id")

    def test_empty_text_yields_no_tokens(self):
        assert tokenize("   \n  ") == ()

    def test_is_deterministic(self):
        text = "The court granted summary judgment under Rule 56."
        assert tokenize(text) == tokenize(text)
