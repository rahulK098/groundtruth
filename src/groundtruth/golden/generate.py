"""Candidate generation. The only module here that talks to Anthropic.

Everything that can be wrong about a proposal -- quote resolution, category
validation, span sizing -- lives in :mod:`groundtruth.golden.proposals` and is
tested without a key. This file is the API call and nothing else, which is why
it is excluded from coverage honestly rather than padded with a mock-only test.

Two details worth knowing:

**The recorded model is the one the API says ran**, taken from the response
rather than from the request. Asking for an alias and recording the alias
would put an unpinned name in the provenance of every candidate -- exactly the
drift that the pinned model revisions elsewhere in this project exist to
prevent.

**Generation proposes; it never labels.** Nothing written here reaches the
golden set without a human decision (ADR-0009).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from groundtruth.golden.candidates import Candidate
from groundtruth.golden.prompts import (
    PROMPT_VERSION,
    PROPOSAL_TOOL,
    SYSTEM_PROMPT,
    user_message,
)
from groundtruth.golden.proposals import RejectedProposal, candidate_from_proposal
from groundtruth.golden.sampling import SampledPassage

#: An alias, resolved to a dated snapshot by the API. The resolved id is what
#: gets recorded, so the alias here never reaches a candidate's provenance.
DEFAULT_MODEL: Final[str] = "claude-sonnet-5"

#: temperature=0 for the usual reason, with the usual caveat: Anthropic has no
#: seed parameter, so this reduces variation rather than eliminating it. What
#: makes the candidate set reproducible is that it is committed, not that the
#: call is deterministic (ADR-0011).
TEMPERATURE: Final[float] = 0.0

MAX_TOKENS: Final[int] = 1200

ProgressCallback = Callable[[int, int], None]


class JudgeExtraNotInstalledError(Exception):
    """The `judge` extra, which provides the Anthropic SDK, is not installed."""


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """What a generation run produced, including what it threw away."""

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


def _client(api_key: str) -> Any:
    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise JudgeExtraNotInstalledError(
            "candidate generation needs the Anthropic SDK, which is not "
            "installed by default:\n\n"
            "    uv sync --frozen --extra dev --extra judge\n\n"
            "It is optional because nothing in the gate path calls it."
        ) from exc
    return anthropic.Anthropic(api_key=api_key)


def _proposal_from_response(response: Any) -> dict[str, Any] | None:
    """Pull the forced tool call out of a response, or None if there is none."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


def generate_candidates(
    passages: Sequence[SampledPassage],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    on_progress: ProgressCallback | None = None,
) -> tuple[tuple[Candidate, ...], GenerationReport]:
    """Propose one candidate per passage.

    Returns the candidates **and** every rejection, because the rejection rate
    is part of the evidence the report quotes.
    """
    client = _client(api_key)
    generated_at = datetime.now(UTC).isoformat()

    candidates: list[Candidate] = []
    rejected: list[RejectedProposal] = []
    resolved_model = model

    for index, passage in enumerate(passages, start=1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=[PROPOSAL_TOOL],
            tool_choice={"type": "tool", "name": PROPOSAL_TOOL["name"]},
            messages=[
                {
                    "role": "user",
                    "content": user_message(
                        passage.doc_id, passage.char_start, passage.char_end, passage.text
                    ),
                }
            ],
        )
        # The concrete dated snapshot, not the alias that was requested.
        resolved_model = getattr(response, "model", model)

        raw = _proposal_from_response(response)
        if raw is None:
            rejected.append(
                RejectedProposal(passage.passage_id, "no-tool-call", "the model returned no proposal")
            )
        else:
            outcome = candidate_from_proposal(
                raw,
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
        model=resolved_model,
        prompt_version=PROMPT_VERSION,
        passages=len(passages),
        accepted=len(candidates),
        rejected=tuple(rejected),
    )
    return tuple(candidates), report
