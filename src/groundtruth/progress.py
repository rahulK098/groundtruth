"""Shared progress-reporting type.

Long-running operations -- a corpus fetch, a cache build -- report progress so
they are distinguishable from a hung process. That distinction matters: a
silent multi-minute operation looks identical to a deadlock, which is exactly
how the first corpus fetch wasted 36 minutes.
"""

from __future__ import annotations

from collections.abc import Callable

#: Called as ``(completed, total)``.
ProgressCallback = Callable[[int, int], None]
