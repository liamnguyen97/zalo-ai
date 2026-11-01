"""Shared state of the pipeline.

INPUT layer (built once per business, cached by the pipeline):
    TABLE DATA  -> `BusinessContext.tables`      (data/loader.py: cleaned DataFrames)
    COLUMNS     -> `BusinessContext.descriptors` (data/descriptor.py: the COLUMN DESCRIPTOR cards)
    + engines over the same data, exact by the rulebook (engines/: sales, items, debt, orders)

`QAState` is the per-question blackboard every stage reads and writes; it also collects the log the competition
requires (/result/logs/<id>.json).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from datamate.data.descriptor import describe_business
from datamate.data.loader import Business, load_business
from datamate.data.onboarding import onboard
from datamate.data.structure import Structure, detect_structure
from datamate.engines.debt import DebtView, build_debt
from datamate.engines.items import ItemView, build_items
from datamate.engines.orders import build_orders
from datamate.engines.purchases import PurchaseView, build_purchases
from datamate.engines.sales import SalesView, build_sales


@dataclass
class Split:
    root: Path
    rulebook: str
    calendar: pd.DataFrame

    @classmethod
    def open(cls, root: str | Path) -> "Split":
        root = Path(root)
        cal = root / "calendar.csv"
        return cls(
            root,
            (root / "rulebook.md").read_text(encoding="utf-8") if (root / "rulebook.md").exists() else "",
            pd.read_csv(cal) if cal.exists() else pd.DataFrame(),
        )


def _safe_build(fn, biz):
    try:
        return fn(biz)
    except Exception:  # noqa: BLE001 - a failed engine must never break answering; the coder still works
        return None


@dataclass
class BusinessContext:
    biz: Business
    descriptors: dict[str, str]
    sales: SalesView | None
    items: ItemView | None
    debt: DebtView | None = None
    orders: pd.DataFrame | None = None     # one row per order, rulebook revenue (engines/orders.py)
    onboarding: dict | None = None         # data/onboarding.py: detected columns + invariants, `trusted` gates the engine reference
    structure: Structure | None = None     # data/structure.py: parent/child tables, document totals, README notes, aliases
    purchases: PurchaseView | None = None  # engines/purchases.py: purchase lines with slip date / status / supplier (rulebook §5)

    @classmethod
    def build(cls, biz_dir: str | Path) -> "BusinessContext":
        biz = load_business(biz_dir)
        sales = _safe_build(build_sales, biz)
        items, debt = _safe_build(build_items, biz), _safe_build(build_debt, biz)
        orders = _safe_build(lambda b: build_orders(b, sales), biz)
        flag = lambda name: os.getenv(name, "1").strip().lower() in {"1", "true", "yes", "on"}   # noqa: E731
        found = detect_structure(biz.tables, biz.readme)
        structure = found if flag("DATAMATE_STRUCTURE_CARDS") else None          # cards: A/B switch
        purchases = _safe_build(lambda b: build_purchases(b, items, found), biz) if flag("DATAMATE_PURCHASE_ENGINE") else None
        report = onboard(biz, sales, items, debt, orders, purchases)
        if structure is not None:
            report["structure"] = structure.summary()
        return cls(biz, describe_business(biz, structure), sales, items, debt, orders, report, structure, purchases)

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return self.biz.tables

    @property
    def readme(self) -> str:
        return self.biz.readme

    @property
    def schema(self) -> str:
        """Full schema card: identical for every question of the business (a stable, cacheable prompt prefix)."""
        return "\n\n".join(self.descriptors.values())


@dataclass
class Period:
    label: str
    start: date
    end: date  # inclusive
    role: str = "main"  # main | compare


@dataclass
class Resolved:
    """The spec's periods as dates; the Coder's `period_*` / `compare_*` variables come from `main` / `compare`."""
    as_of: date
    periods: list[Period] = field(default_factory=list)

    @property
    def main(self) -> Period | None:
        return next((p for p in self.periods if p.role == "main"), None)

    @property
    def compare(self) -> Period | None:
        return next((p for p in self.periods if p.role == "compare"), None)


@dataclass
class QAState:
    q: dict
    ctx: BusinessContext
    log: dict = field(default_factory=dict)
    # Column Selector
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    schema: str = ""
    selection: dict = field(default_factory=dict)
    # Explainer: the checked spec (agents/explainer.Spec), its periods as dates, and the plan for the Coder
    brief: Any = None
    period: Resolved | None = None
    answer_type: str | None = None
    plan: str = ""
    engine_ref: Any = None          # engines/reference.Reference: the Explainer's engine call and its value (or None)
    engine_hint: Any = None         # engines/routing.Hint: advisory engine call derived from the spec (or None)

    @property
    def question(self) -> str:
        return self.q["question"]

    @property
    def as_of(self) -> str:
        return self.q["as_of_date"]
