"""Committed artifact size budget.

ADR-0003 commits to committing the corpus, tokenizer and embedding cache, and
to doing it **without git-LFS** -- because LFS is an extra install, sometimes a
paywall, and either way it breaks the one-command reproduction the whole
project rests on.

That only holds while the artifacts stay small, so the budget is enforced here
rather than remembered. Without this test the cache grows a little at a time
and the no-LFS promise quietly stops being true.
"""

from __future__ import annotations

from pathlib import Path

from groundtruth.paths import project_root

#: Total committed data. Chosen so a clone stays fast on a poor connection.
TOTAL_BUDGET_MB = 20.0

#: No single file. GitHub warns above 50 MB and refuses above 100 MB; 40 MB
#: leaves room to notice before either happens.
PER_FILE_BUDGET_MB = 40.0


def data_files() -> list[Path]:
    root = project_root() / "data"
    if not root.is_dir():
        return []
    # data/raw and data/scratch are gitignored working areas, not artifacts.
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in {"raw", "scratch"} for part in path.parts)
    ]


class TestBudget:
    def test_total_is_within_budget(self):
        files = data_files()
        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        assert total_mb < TOTAL_BUDGET_MB, (
            f"committed data is {total_mb:.2f} MB, over the {TOTAL_BUDGET_MB} MB "
            f"budget from ADR-0003. Either shrink the corpus or revisit the "
            f"no-LFS decision explicitly -- do not just raise this number."
        )

    def test_no_single_file_is_oversized(self):
        for path in data_files():
            size_mb = path.stat().st_size / 1024 / 1024
            assert size_mb < PER_FILE_BUDGET_MB, (
                f"{path.name} is {size_mb:.1f} MB, over the {PER_FILE_BUDGET_MB} MB per-file budget"
            )

    def test_the_artifacts_that_make_reproduction_possible_are_present(self):
        # A passing budget test means nothing if the artifacts are simply
        # absent -- that would be "under budget" and completely broken.
        root = project_root() / "data"
        assert (root / "corpus" / "opinions.jsonl.zst").is_file()
        assert (root / "corpus" / "manifest.json").is_file()
        assert (root / "tokenizers" / "bge-small-en-v1.5" / "tokenizer.json").is_file()
        assert list((root / "cache" / "embeddings").glob("*/vectors.f16.npy"))
