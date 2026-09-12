"""Extracting plain text from opinion HTML.

CourtListener serves opinion bodies as ``plain_text`` for some records and
HTML for others. Both have to reduce to the same kind of string, because
golden-set labels are character offsets into whatever this produces.

Written with the standard library rather than BeautifulSoup or lxml. The input
is a narrow, known shape -- court opinion markup, not the open web -- and a
parser dependency here would be carried by every install, including the gate
path that is deliberately kept tiny.

The bias, as in ``normalize``, is conservative: preserve text, drop only
structure. Block elements become newlines so paragraph boundaries survive,
because collapsing an opinion into one line would merge a holding into the
facts preceding it.
"""

from __future__ import annotations

import html
import re
from typing import Final

#: Dropped entirely, including their content -- it is markup machinery, not
#: opinion text. Everything else keeps its text and loses only its tags.
_DISCARDED_ELEMENTS: Final[re.Pattern[str]] = re.compile(
    r"<(script|style|head|noscript)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_COMMENT: Final[re.Pattern[str]] = re.compile(r"<!--.*?-->", re.DOTALL)

#: Elements whose boundaries are paragraph breaks in a rendered opinion.
_BLOCK_ELEMENT: Final[re.Pattern[str]] = re.compile(
    r"</?(p|div|br|li|ul|ol|tr|table|h[1-6]|blockquote|section|article|pre)\b[^>]*/?>",
    re.IGNORECASE,
)

_ANY_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def looks_like_html(text: str) -> bool:
    """Cheap check for whether a field carries markup.

    Used to decide whether a record's body needs extraction. Deliberately
    crude: a false positive costs one harmless extraction pass, whereas
    running a real parser on every record to find out costs far more.
    """
    return bool(re.search(r"<\s*(p|div|br|span|body|html)\b", text, re.IGNORECASE))


def html_to_text(markup: str) -> str:
    """Reduce opinion markup to plain text, preserving paragraph structure.

    Not a general-purpose HTML renderer. It handles the constructs that appear
    in court opinion bodies and makes no attempt at tables, CSS or layout.
    """
    if not markup:
        return ""

    text = _COMMENT.sub("", markup)
    text = _DISCARDED_ELEMENTS.sub("", text)

    # Block boundaries become newlines BEFORE tags are stripped, or every
    # paragraph in the opinion would be run together.
    text = _BLOCK_ELEMENT.sub("\n", text)
    text = _ANY_TAG.sub("", text)

    # Entities last: unescaping first could introduce a "<" that the tag
    # patterns above would then misread as markup.
    return html.unescape(text)
