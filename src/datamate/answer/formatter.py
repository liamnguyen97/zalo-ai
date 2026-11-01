"""Formatter: pure-code coercion of a raw value into the competition's exact answer format."""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("non-finite number")
    return v


def _num(v: Any, *, percent: bool = False) -> int | float:
    v = _py(v)
    if isinstance(v, bool):
        raise ValueError("bool is not a number")
    if isinstance(v, str):
        v = float(v.replace(",", "").replace("%", "").strip())
    if percent:
        return round(float(v), 2)
    if float(v) == int(float(v)) or abs(float(v)) >= 1e3:  # money (incl. averages of money) is whole đồng
        return int(round(float(v)))  # VND amounts / counts are integers
    return round(float(v), 2)


# a phone number glued to a name by the data ("Chung Mai Châu (0932254793)", "Đoàn Bích Khoa 0914717486"):
# the answer is the name (rulebook §9: names only, no extra labels)
PHONE_TAIL = re.compile(r"\s*[\(\[\-–,/]?\s*(?:\+?84|0)\d[\d .]{7,12}\d\s*[\)\]]?\s*$")


def _clean_name(s: str) -> str:
    out = PHONE_TAIL.sub("", s).strip()
    return out or s


def _cell(v: Any) -> Any:
    v = _py(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _num(v)
    if isinstance(v, str):
        return _clean_name(v)
    return v if isinstance(v, bool) or v is None else str(v)


_MONTHISH = re.compile(r"(\d{4})(?:\.0)?-(\d{1,2})(?:\.0)?(?:-(\d{1,2})(?:\.0)?)?")


def _norm_date_str(s: str) -> str | None:
    """'2025.0-10.0' -> '2025-10'; '2025-9-3' -> '2025-09-03'. None if not date-like."""
    m = _MONTHISH.fullmatch(s.strip())
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{int(mo):02d}" + (f"-{int(d):02d}" if d else "")


def _to_table(raw: Any) -> dict:
    if isinstance(raw, pd.DataFrame):
        return {"columns": [str(c) for c in raw.columns], "rows": raw.values.tolist()}
    if isinstance(raw, dict) and {"columns", "rows"} <= raw.keys():
        return raw
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(r, dict) for r in raw):
        cols = list(raw[0].keys())
        return {"columns": cols, "rows": [[r.get(c) for c in cols] for r in raw]}
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(r, (list, tuple)) for r in raw):
        return {"columns": [f"col{i + 1}" for i in range(len(raw[0]))], "rows": [list(r) for r in raw]}
    raise ValueError("table answer needs columns/rows")


def format_answer(kind: str, raw: Any) -> dict:
    """Return a submission-line fragment {"type":..., "answer":...}. Raises ValueError on bad input."""
    if isinstance(raw, dict) and str(raw.get("type", "")).lower() == "unanswerable":
        return {"type": "unanswerable", "reason": str(raw.get("reason") or "Dữ liệu không đủ để trả lời")}
    if isinstance(raw, str) and raw.upper().startswith("UNANSWERABLE"):
        return {"type": "unanswerable", "reason": raw.partition(":")[2].strip() or "Dữ liệu không đủ để trả lời"}
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and kind not in ("list", "table"):
        raw = raw[0]
    if isinstance(raw, pd.Series) and kind not in ("list", "table"):
        raw = raw.iloc[0]

    if kind == "number":
        return {"type": "number", "answer": _num(raw)}
    if kind == "percent":
        return {"type": "percent", "answer": _num(raw, percent=True)}
    if kind == "boolean":
        raw = _py(raw)
        if isinstance(raw, str):
            raw = raw.strip().lower() in {"true", "có", "co", "yes", "đúng"}
        return {"type": "boolean", "answer": bool(raw)}
    if kind == "date":
        raw = _py(raw)
        s = str(raw).strip()
        norm = _norm_date_str(s)
        if norm:  # also fixes float-ified parts such as '2025.0-10.0'
            return {"type": "date", "answer": norm}
        return {"type": "date", "answer": pd.to_datetime(s).date().isoformat()}
    if kind == "string":
        return {"type": "string", "answer": _clean_name(str(_py(raw)).strip())}
    if kind == "list":
        seq = raw.tolist() if isinstance(raw, pd.Series) else list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        return {"type": "list", "answer": [_cell(x) for x in seq]}
    if kind == "table":
        raw = _to_table(raw)
        return {
            "type": "table",
            "answer": {"columns": [str(c) for c in raw["columns"]], "rows": [[_cell(c) for c in r] for r in raw["rows"]]},
        }
    raise ValueError(f"unknown type {kind}")
