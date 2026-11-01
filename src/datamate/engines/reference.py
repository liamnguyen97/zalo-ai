"""ENGINE REFERENCE: a value computed by the exact engine functions, used as a cross-check on the Coder.

The Explainer may name ONE engine call that computes exactly what the question asks (`Spec.engine_call`, e.g.
`{"function": "debt_repaid", "args": {"start": "2026-07-01", "end": "2026-07-28"}}`). The harness validates it against the
available functions, runs it, shows the Coder the value, and compares the Coder's answer with it. Why: on dev300 the
Coder hand-built sums the engines already compute exactly (`debt_repaid` equals the gold on both debt-repayment questions it
missed by ~1M; `top_items` equals the gold on 3 of 5 top-k questions the Coder ranked itself). The reference is a hint with
one retry: a Coder that still disagrees after seeing it is trusted (the Explainer may have picked the wrong function).
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

from datamate.text import fold

CALLS = ("sales_metrics", "item_qty", "item_revenue", "top_items", "item_price_on", "item_qty_change", "size_qty",
         "debt_total", "debt_by_customer", "debt_customers", "debt_aging", "debt_repaid", "debt_new", "top_customers",
         "purchase_total", "purchase_avg_price", "top_supplier")
DATE_ARGS = ("start", "end", "day")


@dataclass
class Reference:
    call: dict
    value: Any

    @property
    def text(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.call["args"].items())
        return f"{self.call['function']}({args}) -> {repr(self.value)[:400]}"


def clean_call(call: Any) -> dict | None:
    """The Explainer's `engine_call` if it is well formed (known function, scalar args, ISO dates), else None."""
    if not isinstance(call, dict) or call.get("function") not in CALLS or not isinstance(call.get("args", {}), dict):
        return None
    args = call.get("args") or {}
    for k, v in args.items():
        if isinstance(v, (dict, set)):
            return None
        if k in DATE_ARGS and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)[:10]):
            return None
    return {"function": call["function"], "args": dict(args)}


def build_reference(call: Any, tools: dict) -> Reference | None:
    """Run the call on this business's engine functions; None when it is malformed, unavailable or raises."""
    call = clean_call(call)
    fn = tools.get(call["function"]) if call else None
    if fn is None:
        return None
    try:
        inspect.signature(fn).bind(**call["args"])
        value = fn(**call["args"])
    except Exception:  # noqa: BLE001 - a bad reference is dropped, never an error for the question
        return None
    return None if value is None else Reference(call, value)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _names(pairs) -> list[str] | None:
    if isinstance(pairs, (list, tuple)) and pairs and all(isinstance(p, (list, tuple)) and p and isinstance(p[0], str) for p in pairs):
        return [fold(p[0]) for p in pairs]
    return None


def _same_number(a: float, b: float) -> bool:
    return abs(a - b) <= (0.5 if max(abs(a), abs(b)) >= 1000 else 0.011)


def agrees(ref: Reference, raw: Any) -> bool:
    """False only when the Coder's value is of a comparable shape and differs from the reference; shapes that cannot be
    compared (a dict of several metrics, a table of mixed columns) count as agreeing."""
    v = ref.value
    if isinstance(v, dict) and "pct" in v:
        v = v["pct"]
    if (a := _num(v)) is not None:
        b = _num(raw)
        return True if b is None else _same_number(a, b)
    if (names := _names(v)) is not None:
        if isinstance(raw, str):
            return fold(raw) == names[0]
        if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
            got = [fold(str(r[0])) for r in raw["rows"] if r]
        elif isinstance(raw, (list, tuple)) and all(isinstance(x, str) for x in raw):
            got = [fold(x) for x in raw]
        else:
            return True
        return got == names[: len(got)]
    if isinstance(v, (list, tuple)) and v and all(_num(x) is not None for x in v):
        if isinstance(raw, (list, tuple)) and len(raw) == len(v) and all(_num(x) is not None for x in raw):
            return all(_same_number(float(x), float(y)) for x, y in zip(v, raw))
    return True


def reference_note(ref: Reference) -> str:
    """The Coder's task section."""
    return ("# ENGINE REFERENCE (exact by the rulebook)\n"
            f"The explainer picked this engine call as computing exactly what the question asks: `{ref.text}`.\n"
            "Prefer it: assign `answer` from that call (e.g. `answer = <function>(<same arguments>)`, shaped as OUTPUT requires) "
            "instead of rebuilding the number by hand. Differ from it only when the question has a condition the function "
            "cannot express, and print one line saying which.\n\n")
