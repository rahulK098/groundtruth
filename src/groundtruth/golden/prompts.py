"""The candidate-generation prompt, versioned.

``PROMPT_VERSION`` is recorded on every candidate and every pair derived from
one. Editing the prompt without bumping it would make two generation runs
indistinguishable in the record while producing materially different
candidates -- and the report claims the origin breakdown means something.

The instructions below exist to counter the specific failure ADR-0009 names:
LLM-authored queries echo their source passage, which inflates lexical and
dense retrieval alike and makes every configuration look good.
"""

from __future__ import annotations

from typing import Any, Final

PROMPT_VERSION: Final[str] = "golden-candidate-v1"

CATEGORY_GUIDE: Final[str] = """\
factual-lookup        one fact or rule, stated directly in the passage
multi-hop             requires combining two separate statements
negation-exclusion    turns on what does NOT apply, or on an exception
procedural            about posture, standard of review, or process
ambiguous-terminology uses a term whose meaning depends on context\
"""

SYSTEM_PROMPT: Final[str] = f"""\
You are helping build an evaluation set for a legal research retrieval system.
You propose candidate questions. You do not label anything: every proposal you
make is reviewed and accepted, edited or rejected by a human.

Given a passage from a US Supreme Court opinion, propose ONE question that a
practising attorney might actually ask, which this passage answers.

Rules, in order of importance:

1. PARAPHRASE. Do not reuse the passage's distinctive wording. A question
   assembled from the passage's own vocabulary tests whether retrieval can
   find text it was handed, which is worthless. Ask it the way someone who has
   not read this passage would ask it.
2. Quote the MINIMAL span that answers the question -- typically one or two
   sentences. Never quote the whole passage. An oversized span makes the
   question trivially easy to score and biases the comparison.
3. Quote EXACTLY. Every quote must appear character for character in the
   passage, including punctuation and spacing. Quotes that do not match are
   discarded.
4. Do not invent case names, holdings or citations. Everything comes from the
   passage in front of you.
5. If the passage is boilerplate -- a syllabus header, a table of citations, a
   procedural recital with no holding -- say so and propose nothing.

Grade each span you quote:

  3  dispositive: directly states the answer or the controlling rule
  2  relevant: materially supports the answer
  1  partial: related, not sufficient on its own

Categories:

{CATEGORY_GUIDE}
"""

USER_TEMPLATE: Final[str] = """\
Document: {doc_id}
Characters {char_start}-{char_end} of the normalized opinion text.

---
{passage}
---

Propose one question, or decline if this passage is boilerplate.\
"""

#: Forced tool schema rather than free-text JSON. A model asked for JSON in
#: prose eventually returns it wrapped in an explanation, and the parser then
#: has to guess; a forced tool call is schema-checked by the API itself.
PROPOSAL_TOOL: Final[dict[str, Any]] = {
    "name": "propose_candidate",
    "description": (
        "Propose one candidate question with the spans that answer it, or "
        "decline when the passage carries no holding worth retrieving."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decline": {
                "type": "boolean",
                "description": "True when the passage is boilerplate. Omit the other fields.",
            },
            "decline_reason": {"type": "string"},
            "query": {
                "type": "string",
                "description": "The question, paraphrased -- not built from the passage's wording.",
            },
            "category": {
                "type": "string",
                "enum": [
                    "factual-lookup",
                    "multi-hop",
                    "negation-exclusion",
                    "procedural",
                    "ambiguous-terminology",
                ],
            },
            "spans": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {
                            "type": "string",
                            "description": "Exact substring of the passage. Minimal, not the whole passage.",
                        },
                        "gain": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                    "required": ["quote", "gain"],
                },
            },
        },
        "required": [],
    },
}


def user_message(doc_id: str, char_start: int, char_end: int, passage: str) -> str:
    return USER_TEMPLATE.format(
        doc_id=doc_id, char_start=char_start, char_end=char_end, passage=passage
    )
