"""Generic data and date tools for the Explainer agent's lookups (agents/explainer.run_lookups).

They replace hand-written question parsing (abbreviation lists, time-phrase regexes, entity stop-words): instead of
code guessing what "bữa qua", "sp", "tuần rồi" or "dau gio xanh" mean, the agent reads the question and LOOKS THINGS UP
with these tools. Nothing here knows any business type, product or phrase: it searches the data's own values, does
calendar arithmetic, and reads the split's own calendar.csv. Every tool returns short plain text for the agent.
"""
from __future__ import annotations

import difflib
from datetime import date, timedelta

import pandas as pd

from datamate.data.descriptor import describe_column
from datamate.data.structure import resolve_table
from datamate.text import fold_text

WEEKDAYS = ["Thứ Hai (Monday)", "Thứ Ba (Tuesday)", "Thứ Tư (Wednesday)", "Thứ Năm (Thursday)", "Thứ Sáu (Friday)",
            "Thứ Bảy (Saturday)", "Chủ Nhật (Sunday)"]




def _d(x) -> date:
    return pd.Timestamp(x).date()


class DataTools:
    """Bound to one business (tables) and one split (calendar). Methods are exposed to the agent by name."""

    MAX_DISTINCT = 20000

    def __init__(self, tables: dict[str, pd.DataFrame], calendar: pd.DataFrame | None = None, structure=None):
        self.tables = tables
        self.structure = structure        # data/structure.Structure: parent/child links (repair_spec moves misplaced columns)
        self.calendar = calendar if calendar is not None else pd.DataFrame()
        self._values: list[tuple[str, str, str, str]] | None = None  # (table, column, value, folded)

    # ------------------------------------------------------------------ data
    def _index(self) -> list[tuple[str, str, str, str]]:
        if self._values is None:
            out = []
            for t, df in self.tables.items():
                for c in df.columns:
                    s = df[c]
                    if pd.api.types.is_datetime64_any_dtype(s) or s.dtype.kind in "iufb":
                        continue
                    u = s.dropna().astype(str).unique()
                    if len(u) > self.MAX_DISTINCT:
                        continue
                    out += [(t, c, v, fold_text(v)) for v in u if v.strip()]
            self._values = out
        return self._values

    def find(self, text: str, table: str | None = None, column: str | None = None, limit: int = 8) -> list[tuple[float, str, str, str]]:
        """[(score, table, column, value)] of the data's text values that best match `text`
        (accent/case-insensitive, typo-tolerant); score 1.3 = exact match after folding."""
        q = fold_text(text)
        qt = set(q.split())
        scored = []
        for t, c, v, f in self._index():
            if (table and t != table) or (column and c != column):
                continue
            ft = set(f.split())
            overlap = len(qt & ft) / max(len(qt), 1)
            if overlap == 0 and q not in f:
                continue
            ratio = difflib.SequenceMatcher(None, q, f).ratio()
            scored.append((round(0.6 * overlap + 0.4 * ratio + (0.3 if q == f else 0) + (0.1 if q in f else 0), 3), t, c, v))
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for s in scored:
            if s[1:] not in seen:
                seen.add(s[1:])
                out.append(s)
            if len(out) >= limit:
                break
        return out

    def find_values(self, text: str, table: str | None = None, column: str | None = None, limit: int = 8) -> str:
        """Text form of `find`, for an agent: the EXACT spellings stored in the data that match what the user typed.

        A table name the README uses is mapped to the real key. When the asked table has no such value, the other tables
        are searched and the answer says so: biz_109 looked for 'Mì Omachi sườn hầm' in the purchase LINES, which hold only
        the item code, and got a bare "no value" (the name is in the product table)."""
        note = ""
        if table is not None:
            real = resolve_table(table, self.tables)
            if real is None:
                note, table, column = f"(no table {table!r}; searched every table)\n", None, None
            elif real != table:
                note, table = f"(`{table}` is the table `{real}`)\n", real
        hits = self.find(text, table, column, limit)
        if not hits and table is not None:
            hits = self.find(text, None, None, limit)
            if hits:
                note += f"(no value like {text!r} in `{table}`" + (f".`{column}`" if column else "") + "; found in other tables:)\n"
        body = "\n".join(f"{s:.2f}  table={t!r} column={c!r} value={v!r}" for s, t, c, v in hits)
        return note + body if body else note + f"no value resembling {text!r}"

    def describe(self, table: str, column: str | None = None) -> str:
        """A table's columns (name: dtype, range or values), or one column in detail."""
        real = resolve_table(table, self.tables)
        if real is None:
            return f"no table {table!r}; tables: {list(self.tables)}"
        if real != table:                       # the README's name for it (biz_109: `DanhSachNhapHang`)
            return f"(`{table}` is the table `{real}`; use that name)\n" + self.describe(real, column)
        df = self.tables[table]
        if column is None:
            return f"{table}: {len(df)} rows\n" + "\n".join(f"- {c!r}: {describe_column(df[c], c)}" for c in df.columns)
        if column not in df.columns:
            close = difflib.get_close_matches(column, [str(c) for c in df.columns], n=3)
            return f"no column {column!r} in {table}; close: {close}"
        return f"{table}.{column}: {describe_column(df[column], column, max_examples=10)}"

    # ------------------------------------------------------------------ dates
    def date_info(self, day) -> str:
        """Weekday, its Monday-Sunday week, month, quarter, and lunar date / holiday from the split's calendar."""
        d = _d(day)
        mon = d - timedelta(days=d.weekday())
        q = (d.month - 1) // 3 + 1
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        out = [f"{d.isoformat()} = {WEEKDAYS[d.weekday()]}",
               f"week (Mon-Sun): {mon.isoformat()} .. {(mon + timedelta(days=6)).isoformat()}",
               f"month: {d.replace(day=1).isoformat()} .. {(nxt - timedelta(days=1)).isoformat()}",
               f"quarter Q{q}: {date(d.year, 3 * q - 2, 1).isoformat()} .. "
               f"{(date(d.year + (q == 4), (3 * q) % 12 + 1, 1) - timedelta(days=1)).isoformat()}"]
        row = self._cal_row(d)
        if row is not None:
            leap = " (leap month)" if str(row.get("lunar_leap", "")).lower() in ("1", "true") else ""
            hol = f", holiday: {row['holiday']}" if isinstance(row.get("holiday"), str) and row["holiday"].strip() else ""
            out.append(f"lunar: day {row['lunar_day']} month {row['lunar_month']}{leap} year {row['lunar_year']}{hol}")
        return "\n".join(out)

    def shift_date(self, day, days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> str:
        """Date arithmetic; months/years clamp to the last day of a shorter month (31/03 - 1 month = 28/02)."""
        d = pd.Timestamp(_d(day)) + pd.DateOffset(days=days, weeks=weeks, months=months, years=years)
        return d.date().isoformat()

    def lunar_to_solar(self, lunar_year: int, lunar_month: int, lunar_day: int, leap: bool = False) -> str:
        """Solar date of a lunar date, from the split's calendar.csv (e.g. 23 tháng Chạp = lunar month 12, day 23)."""
        cal = self.calendar
        if cal.empty:
            return "no calendar available"
        m = (cal["lunar_year"] == lunar_year) & (cal["lunar_month"] == lunar_month) & (cal["lunar_day"] == lunar_day)
        if "lunar_leap" in cal.columns:
            m &= cal["lunar_leap"].astype(str).str.lower().isin(["1", "true"] if leap else ["0", "false", "nan", ""])
        hit = cal[m]
        return _d(hit["date"].iloc[0]).isoformat() if len(hit) else f"lunar {lunar_day}/{lunar_month}/{lunar_year} not in calendar"

    def _cal_row(self, d: date):
        cal = self.calendar
        if cal.empty or "date" not in cal.columns:
            return None
        if "_d" not in cal.columns:
            cal["_d"] = pd.to_datetime(cal["date"]).dt.date
        hit = cal[cal["_d"] == d]
        return hit.iloc[0] if len(hit) else None

    def functions(self) -> dict:
        """The callables handed to the agent's sandbox."""
        return {"find_values": self.find_values, "describe": self.describe, "date_info": self.date_info,
                "shift_date": self.shift_date, "lunar_to_solar": self.lunar_to_solar}
