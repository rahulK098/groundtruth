"""Reading and writing the golden-set files.

Three files under ``data/golden/``:

``review_log.jsonl``
    **The source of truth.** Append-only, one decision per line, flushed and
    fsynced after every write. Two hours of review is the project's critical
    path (ADR-0009) and losing an hour of it to a crash is not acceptable.

``golden_set.jsonl``
    Materialized from the log. Committed because it is what the harness reads,
    but always regenerable -- so a corrupted golden set is an inconvenience
    rather than a redo of the review.

``candidates.jsonl``
    Every proposal, including the rejected ones. Retaining them is what makes
    the rejection rate evidence and cherry-picking visible in the diff.

JSONL rather than one JSON document because the log is appended to mid-review:
a partial write costs the last line rather than the whole file, and git diffs
it line by line.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import BaseModel, ValidationError

from groundtruth.golden.candidates import Candidate, ReviewDecision, materialize
from groundtruth.golden.models import GoldenPair, GoldenSet

GOLDEN_SET_FILENAME: Final[str] = "golden_set.jsonl"
CANDIDATES_FILENAME: Final[str] = "candidates.jsonl"
REVIEW_LOG_FILENAME: Final[str] = "review_log.jsonl"

ModelT = TypeVar("ModelT", bound=BaseModel)


class GoldenStoreError(Exception):
    """A golden-set file could not be read or written."""


def _to_line(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _read_models(path: Path, model: type[ModelT], *, kind: str) -> tuple[ModelT, ...]:
    if not path.is_file():
        return ()

    out: list[ModelT] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            # Naming the file and line matters more here than anywhere else:
            # a hand-edited review log is how this breaks in practice.
            raise GoldenStoreError(f"{path.name} line {lineno} is not a valid {kind}: {exc}") from exc
    return tuple(out)


def _append(path: Path, lines: Iterable[str]) -> None:
    """Append durably.

    ``flush`` then ``fsync`` on every append. The cost is a millisecond per
    decision against a reviewer's several seconds of thought, and it is what
    makes "the log survives a crash" true rather than probable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# --- review log -------------------------------------------------------------


def append_decision(decision: ReviewDecision, directory: Path) -> None:
    """Record one review decision. Never overwrites an earlier one."""
    _append(directory / REVIEW_LOG_FILENAME, [_to_line(decision)])


def read_review_log(directory: Path) -> tuple[ReviewDecision, ...]:
    return _read_models(
        directory / REVIEW_LOG_FILENAME, ReviewDecision, kind="review decision"
    )


# --- candidates -------------------------------------------------------------


def append_candidates(candidates: Sequence[Candidate], directory: Path) -> None:
    _append(directory / CANDIDATES_FILENAME, (_to_line(c) for c in candidates))


def read_candidates(directory: Path) -> tuple[Candidate, ...]:
    return _read_models(directory / CANDIDATES_FILENAME, Candidate, kind="candidate")


# --- golden set -------------------------------------------------------------


def write_golden_set(golden: GoldenSet, directory: Path) -> None:
    """Write the materialized set, replacing whatever was there.

    Sorted and key-sorted, so the same set always produces byte-identical
    output and a re-materialization with no review activity shows an empty
    git diff rather than noise.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [_to_line(pair) for pair in golden.pairs]
    (directory / GOLDEN_SET_FILENAME).write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8", newline="\n"
    )


def read_golden_set(directory: Path) -> GoldenSet:
    pairs = _read_models(directory / GOLDEN_SET_FILENAME, GoldenPair, kind="golden pair")
    try:
        return GoldenSet(pairs=pairs)
    except ValidationError as exc:
        raise GoldenStoreError(f"{GOLDEN_SET_FILENAME} is not a valid golden set: {exc}") from exc


def rebuild_golden_set(directory: Path) -> GoldenSet:
    """Replay the review log and write the golden set it describes."""
    golden = materialize(read_review_log(directory))
    write_golden_set(golden, directory)
    return golden


def review_summary(directory: Path) -> dict[str, Any]:
    """Counts the report quotes verbatim.

    "Generated 150, accepted 62, edited 38, rejected 50" is a far stronger
    claim than "here are 100 pairs", and it is only available because the log
    keeps every decision including the reversed ones.
    """
    decisions = read_review_log(directory)
    latest = {decision.subject: decision for decision in decisions}

    counts = {"accept": 0, "edit": 0, "reject": 0, "skip": 0}
    for decision in latest.values():
        counts[decision.action] += 1

    return {
        "candidates_generated": len(read_candidates(directory)),
        "decisions_recorded": len(decisions),
        "subjects_decided": len(latest),
        **counts,
        "superseded": len(decisions) - len(latest),
    }
