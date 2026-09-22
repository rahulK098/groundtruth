"""Candidate generation.

The only module here that talks to a model -- and it does so through the
vendor-neutral provider protocol in :mod:`groundtruth.llm`, so Azure, Groq,
Gemini, a local Ollama or Anthropic all drive it unchanged.

Everything that can be wrong about a proposal -- quote resolution, category
validation, span sizing -- lives in :mod:`groundtruth.golden.proposals` and is
tested without a key. This file is the call loop and nothing else.

Three details worth knowing:

**The recorded model is the one the API says ran**, taken from the response
rather than from the request. Asking for an alias and recording the alias
would put an unpinned name in the provenance of every candidate -- exactly the
drift that the pinned model revisions elsewhere in this project exist to
prevent.

**One provider per run, chosen explicitly.** There is no fallback chain: a run
answered by whichever vendor happened to be reachable would not be
reproducible, and nothing in the record would mark the mixture as unintended.

**Generation proposes; it never labels.** Nothing written here reaches the
golden set without a human decision (ADR-0009).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from groundtruth.golden.candidates import Candidate
from groundtruth.golden.prompts import (
    PROMPT_VERSION,
    PROPOSAL_TOOL_SPEC,
    SYSTEM_PROMPT,
    user_message,
)
from groundtruth.golden.proposals import RejectedProposal, candidate_from_proposal
from groundtruth.golden.sampling import SampledPassage
from groundtruth.llm.models import ChatProvider, LLMError

#: temperature=0 for the usual reason, with the usual caveat: no vendor here
#: offers a seed, so this reduces variation rather than eliminating it. What
#: makes the candidate set reproducible is that it is committed, not that the
#: call is deterministic (ADR-0011).
TEMPERATURE: Final[float] = 0.0

MAX_TOKENS: Final[int] = 1200

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """What a generation run produced, including what it threw away."""

    provider: str
    model: str
    prompt_version: str
    passages: int
    accepted: int
    rejected: tuple[RejectedProposal, ...]

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.code] = counts.get(rejection.code, 0) + 1
        return counts


def generate_candidates(
    passages: Sequence[SampledPassage],
    provider: ChatProvider,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[tuple[Candidate, ...], GenerationReport]:
    """Propose one candidate per passage.

    Returns the candidates **and** every rejection, because the rejection rate
    is part of the evidence the report quotes.
    """
    generated_at = datetime.now(UTC).isoformat()

    candidates: list[Candidate] = []
    rejected: list[RejectedProposal] = []
    resolved_model = provider.model

    for index, passage in enumerate(passages, start=1):
        try:
            completion = provider.complete(
                system=SYSTEM_PROMPT,
                user=user_message(
                    passage.doc_id, passage.char_start, passage.char_end, passage.text
                ),
                tool=PROPOSAL_TOOL_SPEC,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        except LLMError as exc:
            # Recorded rather than raised: one passage that trips a content
            # filter or a transient 500 should not discard the 80 candidates
            # already generated. A systemic failure shows up as a run whose
            # rejection counts are almost entirely this code.
            rejected.append(RejectedProposal(passage.passage_id, "request-failed", str(exc)))
            if on_progress is not None:
                on_progress(index, len(passages))
            continue

        # The concrete snapshot the API reports, not the alias requested.
        resolved_model = completion.model

        if completion.tool_input is None:
            rejected.append(
                RejectedProposal(
                    passage.passage_id, "no-tool-call", "the model returned no proposal"
                )
            )
        else:
            outcome = candidate_from_proposal(
                completion.tool_input,
                passage,
                candidate_id=f"c-{index:04d}",
                generator_model=resolved_model,
                prompt_version=PROMPT_VERSION,
                generated_at=generated_at,
            )
            if isinstance(outcome, RejectedProposal):
                rejected.append(outcome)
            else:
                candidates.append(outcome)

        if on_progress is not None:
            on_progress(index, len(passages))

    report = GenerationReport(
        provider=provider.name,
        model=resolved_model,
        prompt_version=PROMPT_VERSION,
        passages=len(passages),
        accepted=len(candidates),
        rejected=tuple(rejected),
    )
    return tuple(candidates), report
