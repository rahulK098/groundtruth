"""ADR-0008's layering test: the core never imports FastAPI.

Run in a clean interpreter, because in this one the service tests have
already imported fastapi -- an in-process check would pass or fail depending
on test order, which would make it worthless.
"""

from __future__ import annotations

import subprocess
import sys

CORE_MODULES = (
    "groundtruth.retrieval.pipeline",
    "groundtruth.retrieval.build",
    "groundtruth.scoring.evaluate",
    "groundtruth.gate.compare",
    "groundtruth.rerank.cache",
    "groundtruth.index.bm25",
    "groundtruth.index.numpy_vector",
    "groundtruth.cli.main",
)

FORBIDDEN = ("fastapi", "starlette", "uvicorn", "torch", "sentence_transformers", "psycopg")


def test_the_core_imports_no_web_framework_model_library_or_database_driver():
    probe = (
        "import sys\n"
        + "".join(f"import {module}\n" for module in CORE_MODULES)
        + f"loaded = sorted(m for m in {FORBIDDEN!r} if m in sys.modules)\n"
        + "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    loaded = [name for name in result.stdout.strip().split(",") if name]
    assert loaded == [], (
        f"the core pulled in {loaded}. The gate environment installs none of them, so "
        f"this import would break `pytest -m gate` there (ADR-0003, ADR-0008)."
    )
