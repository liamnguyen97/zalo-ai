"""Memory cap for the harness process.

The Coder's code runs in-process, so one runaway pandas block (a many-to-many merge, a cross join) could allocate tens of
GB, make the kernel OOM-kill the process and, on the shared box, the vLLM engine next to it (dev300 run, 10-30). With an
address-space limit the oversized allocation raises MemoryError inside that block instead; the Runner reports it to the
Coder as an error and the turn goes on.
"""
from __future__ import annotations

import os
import sys

OOM_HINT = (
    "MEMORY: this code block asked for too much memory and was stopped (usually a many-to-many merge/join or a cross product "
    "producing tens of millions of rows). Filter by period/status BEFORE merging, aggregate (groupby) each table to one row per "
    "key and only then merge, or use `sales_orders` / `item_lines` / the engine functions instead of joining tables yourself.")


def limit_gb() -> float:
    return float(os.getenv("DATAMATE_MEM_LIMIT_GB", "16"))


def apply_limit(gb: float | None = None) -> float:
    """Cap the process's address space at `gb` GB (0 = no cap). Returns the cap applied, 0 if none (not Linux, or refused)."""
    gb = limit_gb() if gb is None else gb
    if gb <= 0 or not sys.platform.startswith("linux"):   # macOS does not enforce RLIMIT_AS
        return 0.0
    import resource
    cap = int(gb * 1024 ** 3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        cap = min(cap, hard)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    except (ValueError, OSError):
        return 0.0
    return cap / 1024 ** 3


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or "MemoryError" in str(exc) or "Unable to allocate" in str(exc)
