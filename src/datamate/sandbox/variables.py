"""RUNNER inputs: what the coder's sandbox sees (tables, calendar, the spec's periods, `sales_orders`, `item_lines` and
the engine functions of engines/functions.py)."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from datamate.engines.functions import build_tools
from datamate.engines.items import with_item_attrs
from datamate.text import fold


# `pd.api.types.*` is "Forbidden access to module: pandas" in the sandbox (41 failed blocks on dev300 a1): the guard rewrites
# the call to the bare name and the sandbox provides these functions
DTYPE_HELPERS = ("is_datetime64_any_dtype", "is_numeric_dtype", "is_string_dtype", "is_object_dtype", "is_integer_dtype",
                 "is_float_dtype")


def _ts(d):
    return pd.Timestamp(d) if d is not None else None


def _end_of_day(d):
    # inclusive END OF DAY: both `col <= period_end` and `col < period_end` keep the last day (the model kept dropping it)
    return pd.Timestamp(d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) if d is not None else None


class Runner:
    """Builds the coder's sandbox variables. Engine functions and enriched item lines are built once per business."""

    def __init__(self, calendar: pd.DataFrame):
        self.calendar = calendar
        self._tools: dict[int, dict] = {}
        self._lines: dict[int, pd.DataFrame | None] = {}

    def tools(self, ctx) -> dict:
        key = id(ctx)
        if key not in self._tools:
            self._tools[key] = build_tools(getattr(ctx, "sales", None), getattr(ctx, "items", None), getattr(ctx, "debt", None),
                                           getattr(ctx, "orders", None), getattr(ctx, "purchases", None))
        return self._tools[key]

    def item_lines(self, ctx) -> pd.DataFrame | None:
        """item_lines with the product tables' columns (group, prescription flag...), built once per business."""
        key = id(ctx)
        if key not in self._lines:
            items = getattr(ctx, "items", None)
            df = with_item_attrs(items.lines, ctx.tables) if items is not None else None
            if df is not None:
                df.attrs["source_table"] = items.table      # the guard tells sales lines from other tables with it
            self._lines[key] = df
        return self._lines[key]

    def variables(self, st) -> dict:
        main = st.period.main if st.period else None
        cmp_ = st.period.compare if st.period else None
        orders = getattr(st.ctx, "orders", None)
        sales_orders = orders.copy(deep=False) if orders is not None else None
        item_lines = self.item_lines(st.ctx)
        tables = {k: v.copy(deep=False) for k, v in st.tables.items()}
        # also reachable as tables['sales_orders'] / tables['item_lines']: the Coder looked for them there, found nothing and
        # answered UNANSWERABLE (dev300 1885ab, f19c4e); the same objects as the variables below
        for name, df in (("sales_orders", sales_orders), ("item_lines", item_lines)):
            if df is not None and name not in tables:
                tables[name] = df
        return {
            # shallow copies + copy-on-write: `df['Ngày'] = ...` inside the sandbox stays local to this run (seen: the agent
            # converted a shared column in place, which would leak into every concurrent question of the business)
            "tables": tables, "calendar": self.calendar.copy(deep=False),
            "as_of_date": st.as_of,
            "period_start": _ts(main.start) if main else None, "period_end": _end_of_day(main.end) if main else None,
            "compare_start": _ts(cmp_.start) if cmp_ else None, "compare_end": _end_of_day(cmp_.end) if cmp_ else None,
            "item_lines": item_lines, "sales_orders": sales_orders,
            "fold": fold, **{n: getattr(pdt, n) for n in DTYPE_HELPERS},
            **self.tools(st.ctx),
        }
