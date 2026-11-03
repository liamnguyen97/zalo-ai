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
from datamate.text import fold, fold_text, norm

LIST_TYPES = ("list", "string", "table")
# The gates read the spec's STRUCTURED fields and the ORIGINAL question (its own words, read in code), not the free-text rewrite, which
# is the model's English. An English word of the rewrite used to be the gate ("customer" opened `top_customers`, "stock" / "average"...
# closed `item_qty`): "3 khách còn nợ nhiều nhất" got the REVENUE ranking `top_customers` as its hint in both turns (dev300
# q_4333ad0589a1), although its spec named `debt_by_customer` (not available in that shop) and cited §4 Công nợ.
# The rulebook sections whose numbers are not the sales of a period (debt; stock, purchases, cost), and the sales ones (orders and
# revenue, rankings), on their headings (Spec.rule_sections). A spec citing only the former gets no sales hint. Citing one of them is not
# enough: the model cites §5 "Nhập trừ bán (quy về cùng đơn vị)" for the unit conversion of a sales question (dev300: 3 right
# `item_qty` hints on "bán được bao nhiêu kg Nếp cái hoa vàng" cite §1 and §5)
OTHER_SECTIONS = re.compile(r"\bcong no\b|\bkho\b|\bnhap hang\b|\bgia von\b")
SALES_SECTIONS = re.compile(r"\bdoanh thu\b|\bxep hang\b")
# The question asks a balance, a purchase or an average, not what was sold in the period ("tồn kho ... cuối ngày", "trong kho còn",
# "số nhập trừ số bán", "khách còn nợ"); read with the spec's entity values taken out ("Tôn lạnh" is a product)
NOT_SOLD = re.compile(r"\bton\b|\btrong kho\b|\bcon hang\b|\bhet hang\b|\bcon lai\b|\bno\b|\bnhap\b|\btrung binh\b|\bbinh quan\b")
# ... or money, when the spec still says "quantity" (`item_qty` counts goods)
MONEY = re.compile(r"\btien\b|\bdoanh thu\b|\bgia\b|\blai gop\b|\bvon\b|\bchi phi\b")
# a ranking of customers: "khách" (not a customer group), "ai mua", "ai chi"
CUSTOMERS = re.compile(r"(?<!nhom )(?<!loai )\bkhach\b|\bai (?:mua|chi)\b")


def question_words(question: str | None, entities: list[dict]) -> str:
    """The question normalised (text.norm), the spec's entity values taken out: what the question says around its names."""
    q = norm(question or "")
    for v in sorted({norm(str(e.get("value") or "")) for e in entities}, key=len, reverse=True):
        if v:
            q = re.sub(rf"(?<!\w){re.escape(v)}(?!\w)", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def off_route(spec, question: str | None, entities: list[dict]) -> str:
    """Why no sales hint fits this spec ("" = one may): the spec cites the debt / stock sections and no sales one, or the question asks
    a balance, a purchase or an average."""
    secs = [fold_text(x) for x in getattr(spec, "rule_sections", None) or []]
    other = [s for s in secs if OTHER_SECTIONS.search(s)]
    if other and not any(SALES_SECTIONS.search(s) for s in secs):
        return f"the spec cites rulebook section {other[0]!r} and no sales section"
    m = NOT_SOLD.search(question_words(question, entities))
    return f"the question says {m.group(0)!r}" if m else ""


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
    if getattr(spec, "status", "") == "fallback" or (question and fold_text(rewrite) == fold_text(question)):
        return None         # the Explainer's fallback copied the question back: no usable spec, every wrong hint came from here
    all_ents = [e for e in (spec.entities or []) if isinstance(e, dict)]
    ents = _without_base_units(all_ents, tools)
    items = [e for e in ents if e.get("role") == "item"]
    s, e = _iso(start), _iso(end)
    atype, measure, k = spec.answer_type, spec.measure, spec.k
    if off_route(spec, question, all_ents):
        return None
    words = question_words(question, all_ents)
    call = getattr(spec, "engine_call", None)
    named = call.get("function") if isinstance(call, dict) else None     # the Explainer's own choice, even when it could not run

    def run(fn: str, **args) -> Hint | None:
        f = tools.get(fn)
        if f is None or (named and named != fn):
            return None
        try:
            return Hint({"function": fn, "args": args}, f(**args), "")
        except Exception:  # noqa: BLE001 - a hint that cannot be computed is simply not given
            return None

    if len(ents) == 1 and len(items) == 1 and atype == "number" and measure == "quantity" and not MONEY.search(words):
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
    if not ents and atype in LIST_TYPES and measure == "money" and k and k >= 1 and CUSTOMERS.search(words):
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
