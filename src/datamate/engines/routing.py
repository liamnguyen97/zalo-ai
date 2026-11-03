"""ENGINE HINTS: tell the Coder which exact engine function fits a question, derived from the spec's structured fields.

The Explainer may name an `engine_call` itself (engines/reference.py), but the 9B model leaves it null on exactly the standard
shapes: on dev300 b1 the engines equal the gold on the Nếp cái kg questions (4/4 `item_qty`), size M/L counts (`size_qty`), an
average order (`sales_metrics`) and top-k items (3 of 5), yet none of those questions had an engine_call and the Coder rebuilt the
numbers by hand and missed. This module looks at the checked spec (measure, answer type, entities, k, periods), picks at most one
engine call that fits, runs it, and the Coder is shown "this function returns X for this question" as an ADVISORY hint:

- it is a hint, not an override: no verdict depends on it, a question with a condition the function cannot express (channel,
  staff, payment, shift...) is told to use `sales_orders` / `item_lines` instead;
- it is shown only for shops whose engines passed the onboarding invariants (data/onboarding.py);
- whether the final answer agreed with it is logged (`log["routing"]`), so the hint's precision is measured, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import re

from datamate.engines.reference import Reference, agrees
from datamate.text import fold, fold_text

LIST_TYPES = ("list", "string", "table")
# a rewrite that talks about these is not a "quantity sold" question, whatever the spec's measure says (dev300: "ending inventory")
NOT_QUANTITY = re.compile(r"revenue|money|price|cost|stock|inventory|on hand|average|\bvnd\b|sales amount|purchase|import|profit", re.I)


@dataclass
class Hint:
    call: dict
    value: Any
    note: str

    @property
    def text(self) -> str:
        return Reference(self.call, self.value).text

    def agrees_with(self, raw: Any) -> bool:
        return agrees(Reference(self.call, self.value), raw)


def _iso(ts) -> str:
    return str(ts)[:10]


def _without_base_units(ents: list[dict], tools: dict) -> list[dict]:
    """The entities without the ones that are the BASE unit of another (item) entity: "bao nhiêu cái Bánh tiramisu" gave the spec
    `Bánh tiramisu` and `cái` (column `Đơn vị tính`, role item), so no hint was given (dev300 q_1885ab61433f: the hint is 89 =
    the gold, the Coder said 83). The engine counts in base units, so that unit is no condition; another unit ("thùng" when the
    base unit is "lon") stays an entity, and there is no hint."""
    base = tools.get("item_base_unit")
    if base is None or len(ents) < 2:
        return ents
    units = set()
    for e in ents:
        if e.get("role") != "item":
            continue
        try:
            units.add(fold(str(base(e.get("value")))))
        except Exception:  # noqa: BLE001 - not an item of the data: it has no base unit
            continue
    units.discard("")
    units.discard("none")
    return [e for e in ents if fold(str(e.get("value") or "")) not in units]


def route(spec, tools: dict, start, end, question: str | None = None, compare: bool = False) -> Hint | None:
    """The one engine call that fits this spec, run on [start, end], or None. `start`/`end`: the main period (None = no period).

    Measured offline on the 300 dev300 specs against the gold (hint right / Coder right / hint right and Coder not / hint wrong
    and Coder right): item_qty 16/16, 9/16, 7, 0; top_items 9/10, 8/10, 1, 0; top_customers 2/2. Two shapes were dropped because
    they were worse than the Coder they guide: whole-shop `sales_metrics` (64/91 right against the Coder's 88/91: the spec cannot
    see conditions such as cancelled, channel or shift) and `item_revenue` (11/17 against 16/17)."""
    if spec is None or start is None or end is None or compare or len(getattr(spec, "periods", None) or []) != 1:
        return None
    rewrite = (spec.rewritten_question or "")
    if question and fold_text(rewrite) == fold_text(question):
        return None         # the Explainer's fallback copied the question back: no usable spec, every wrong hint came from here
    ents = _without_base_units([e for e in (spec.entities or []) if isinstance(e, dict)], tools)
    items = [e for e in ents if e.get("role") == "item"]
    s, e = _iso(start), _iso(end)
    atype, measure, k = spec.answer_type, spec.measure, spec.k

    def run(fn: str, **args) -> Hint | None:
        f = tools.get(fn)
        if f is None:
            return None
        try:
            return Hint({"function": fn, "args": args}, f(**args), "")
        except Exception:  # noqa: BLE001 - a hint that cannot be computed is simply not given
            return None

    if len(ents) == 1 and len(items) == 1 and atype == "number" and measure == "quantity" and not NOT_QUANTITY.search(rewrite):
        name = items[0].get("value")
        h = run("item_qty", item=name, start=s, end=e)
        if h is not None and "item_base_unit" in tools:
            try:
                h.note = f"in the item's BASE unit ({tools['item_base_unit'](name)}); convert yourself if the question asks for another unit"
            except Exception:  # noqa: BLE001
                pass
        return h
    if not ents and atype in LIST_TYPES and measure == "quantity" and k and k >= 1:
        h = run("top_items", k=int(k), start=s, end=e, by="qty")
        if h is not None:
            h.note = "ranked by quantity in base units, ties by code (rulebook §3)"
        return h
    if not ents and atype in LIST_TYPES and measure == "money" and k and k >= 1 and "customer" in (spec.rewritten_question or "").lower():
        h = run("top_customers", k=int(k), start=s, end=e)
        if h is not None:
            h.note = "ranked by revenue, walk-in customers excluded (rulebook §3)"
        return h
    return None


def hint_note(h: Hint) -> str:
    """The Coder's task section."""
    return ("# ENGINE HINT (advisory)\n"
            f"An exact engine function fits this question: `{h.text}`" + (f" ({h.note})" if h.note else "") + ".\n"
            "Prefer it when the question has no condition beyond the item / period (no channel, staff, payment method, shift, size or "
            "customer group); otherwise compute from `sales_orders` / `item_lines`. If your result differs from it, print one line saying why.\n\n")
