"""ENGINE REFERENCE: a value computed by the exact engine functions, used as a cross-check on the Coder.

The Explainer may name ONE engine call that computes exactly what the question asks (`Spec.engine_call`, e.g.
`{"function": "debt_repaid", "args": {"start": "2026-07-01", "end": "2026-07-28"}}`). The harness validates it against the
available functions, runs it, shows the Coder the value, and compares the Coder's answer with it. Why: on dev300 the
Coder hand-built sums the engines already compute exactly (`debt_repaid` equals the gold on both debt-repayment questions it
missed by ~1M; `top_items` equals the gold on 3 of 5 top-k questions the Coder ranked itself). The reference is a hint with
one retry: a Coder that still disagrees after seeing it is trusted (the Explainer may have picked the wrong function).

A call with an argument the function does not take is REPAIRED when the argument is a formality, else dropped (the reason is
logged). dev300 run 2b4ca48e19: 120 of 242 calls were dropped whole, 64 for `sales_metrics(..., metric=...)`, with no trace; the
Coder then rebuilt "how many orders today" by hand (0, the engine said 66). Formalities: `valid_only=True` (every engine counts
valid orders only), a `metric` that names a key of the dict the function returns (`order_count` -> `['orders']`), a `unit` that is
the item's base unit (the unit the engine counts in). Anything else is a condition the function cannot express (`shift`, `channel`,
`metric='cancelled_orders'`...): without it the function gives the whole shop's figure. A repaired call is kept only for a plain
number answer: a revenue is not the answer of a percent-change question.
"""
from __future__ import annotations

import inspect
import numbers
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from datamate.text import fold

# `metric` values that name a key of the dict a function returns (sales_metrics): any other metric drops the call
METRIC_KEYS = {"orders": "orders", "order_count": "orders", "n_orders": "orders", "num_orders": "orders", "total_orders": "orders",
               "number_of_orders": "orders", "revenue": "revenue", "total_revenue": "revenue", "avg_order": "avg_order",
               "avg_order_value": "avg_order", "average_order_value": "avg_order", "avg_revenue_per_order": "avg_order", "aov": "avg_order"}
CALLS = ("sales_metrics", "item_qty", "item_revenue", "top_items", "item_price_on", "item_qty_change", "size_qty",
         "debt_total", "debt_by_customer", "debt_customers", "debt_aging", "debt_repaid", "debt_new", "top_customers",
         "purchase_total", "purchase_avg_price", "top_supplier", "cancel_rate")
DATE_ARGS = ("start", "end", "day")


@dataclass
class Reference:
    call: dict
    value: Any
    key: str | None = None      # the key of the dict the function returns that the value is (a repaired `metric`), else None

    @property
    def expr(self) -> str:
        """The call as the Coder should write it, e.g. `sales_metrics(start='...', end='...')['orders']`."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.call["args"].items())
        return f"{self.call['function']}({args})" + (f"[{self.key!r}]" if self.key else "")

    @property
    def text(self) -> str:
        return f"{self.expr} -> {repr(self.value)[:400]}"


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


def _is_base_unit(unit: Any, args: dict, tools: dict) -> bool:
    base = tools.get("item_base_unit")
    if base is None or not args.get("item") or not str(unit).strip():
        return False
    try:
        return fold(str(base(args["item"]))) == fold(str(unit))
    except Exception:  # noqa: BLE001 - an unknown item: not provably the base unit
        return False


def resolve_reference(call: Any, tools: dict, answer_type: str | None = None) -> tuple[Reference | None, str]:
    """(reference, "") or (None, why the call was dropped). Arguments the function takes pass unchanged; the others are repaired
    when they are formalities (module doc), else the call is dropped. A repaired call needs `answer_type` "number"."""
    call = clean_call(call)
    if call is None:
        return None, "malformed call (unknown function, a list / dict argument or a date that is not YYYY-MM-DD)"
    fn = tools.get(call["function"])
    if fn is None:
        return None, f"{call['function']} is not available for this shop"
    params = inspect.signature(fn).parameters
    open_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    args, key, repaired = {}, None, []
    for k, v in call["args"].items():
        if k in params or open_kw:
            args[k] = v
        elif k == "valid_only" and v is True:
            repaired.append(k)
        elif k == "metric" and str(v).strip().lower() in METRIC_KEYS:
            key = METRIC_KEYS[str(v).strip().lower()]
            repaired.append(k)
        elif k == "unit" and _is_base_unit(v, call["args"], tools):
            repaired.append(k)
        else:
            return None, f"argument {k}={v!r}: {call['function']} does not take it (a condition it cannot express)"
    if repaired and answer_type != "number":
        return None, f"repaired call ({', '.join(repaired)} removed) is used only for a number answer; the spec asks for {answer_type!r}"
    try:
        inspect.signature(fn).bind(**args)
        value = fn(**args)
    except Exception as e:  # noqa: BLE001 - a bad reference is dropped, never an error for the question
        return None, f"the call raised {type(e).__name__}: {e}"[:300]
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            return None, f"metric {key!r} is not a key of what {call['function']} returns"
        value = value[key]
    if value is None:
        return None, "the call returned None"
    if repaired and _real(value) is None:
        return None, "a repaired call must give one number"
    return Reference({"function": call["function"], "args": args}, value, key), ""


def build_reference(call: Any, tools: dict, answer_type: str | None = None) -> Reference | None:
    """Run the call on this business's engine functions; None when it is malformed, unavailable, raises, or has an argument that
    cannot be repaired (see resolve_reference)."""
    return resolve_reference(call, tools, answer_type)[0]


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


def _real(x) -> float | None:
    """A plain or numpy number as float (numpy integers are not `int`); None for anything else, booleans included."""
    return float(x) if isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_)) else None


def confirms(ref: Reference, raw: Any) -> bool:
    """True only when the value has a shape the reference can be compared with AND equals it. `agrees` lets through the shapes it
    cannot compare (it decides a doubt); this decides that no judge is needed, so anything unsure is False. A dict of several
    metrics (`sales_metrics`) never confirms: the whole-shop figures ignore conditions such as channel, shift or cancelled orders
    (as a routing hint they were right on 64 of 91 dev300 questions against the Coder's 88)."""
    v = ref.value
    if isinstance(v, dict) and "pct" in v:
        v = v["pct"]
    if (a := _real(v)) is not None:
        b = _real(raw)
        return b is not None and _same_number(a, b)
    if (names := _names(v)) is not None:
        if isinstance(raw, str):
            return len(names) == 1 and fold(raw) == names[0]
        if isinstance(raw, (list, tuple)) and raw and all(isinstance(x, str) for x in raw):
            return [fold(x) for x in raw] == names
        return False
    if isinstance(v, (list, tuple)) and v and all(_real(x) is not None for x in v):
        return (isinstance(raw, (list, tuple)) and len(raw) == len(v) and all(_real(x) is not None for x in raw)
                and all(_same_number(_real(x), _real(y)) for x, y in zip(v, raw)))
    return False


def reference_note(ref: Reference) -> str:
    """The Coder's task section."""
    return ("# ENGINE REFERENCE (exact by the rulebook)\n"
            f"The explainer picked this engine call as computing exactly what the question asks: `{ref.text}`.\n"
            "Prefer it: assign `answer` from that call (e.g. `answer = <function>(<same arguments>)`, shaped as OUTPUT requires) "
            "instead of rebuilding the number by hand. Differ from it only when the question has a condition the function "
            "cannot express, and print one line saying which.\n"
            + (f"The function returns a dict: the value is its `[{ref.key!r}]`.\n" if ref.key else "") + "\n")
