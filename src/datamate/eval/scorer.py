"""Local clone of the official scorer (README "Evaluation") for Penalized Accuracy."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def norm_str(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if (ch.isalpha() or ch.isdigit() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def _num_eq(a, b, kind: str = "number") -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return round(a, 2) == round(b, 2) if kind in ("percent", "number") else a == b


def _scalar_eq(a, b, alias=()) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return _num_eq(a, b)
    forms = {norm_str(b), *(norm_str(x) for x in alias)}
    return norm_str(a) in forms


def is_correct(pred: dict, gold: dict) -> bool:
    if pred.get("type") != gold.get("type"):
        return False
    t, p, g = gold["type"], pred.get("answer"), gold.get("answer")
    alias = gold.get("alias") or []
    if t == "unanswerable":
        return True
    if t in ("number", "percent"):
        return _num_eq(p, g, t)
    if t in ("string", "date", "boolean"):
        return (p is g) if t == "boolean" else _scalar_eq(p, g, alias) if t == "string" else str(p) == str(g)
    if t == "list":
        return isinstance(p, list) and len(p) == len(g) and all(_scalar_eq(x, y) for x, y in zip(p, g))
    if t == "table":
        try:
            pc, gc = p["columns"], g["columns"]
            prow, grow = p["rows"], g["rows"]
        except (TypeError, KeyError):
            return False
        if len(pc) != len(gc) or len(prow) != len(grow):
            return False
        # columns are matched by normalized name, else by position
        idx = [pc.index(c) if c in pc else i for i, c in enumerate(gc)] if {norm_str(c) for c in pc} == {norm_str(c) for c in gc} else list(range(len(gc)))
        def key(row):
            return tuple(round(float(c), 2) if isinstance(c, (int, float)) and not isinstance(c, bool) else norm_str(c) for c in row)
        return Counter(key([r[i] for i in idx]) for r in prow) == Counter(key(r) for r in grow)
    return False


def load_jsonl(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def score(pred_path: str | Path, gold_path: str | Path, ids: set[str] | None = None) -> dict:
    preds, gold = load_jsonl(pred_path), load_jsonl(gold_path)
    ids = ids if ids is not None else set(gold)
    c = w = a = 0
    by_type: dict[str, Counter] = {}
    for i in ids:
        g, p = gold[i], preds.get(i, {"type": "abstain"})
        bucket = by_type.setdefault(g["type"], Counter())
        if p.get("type") == "abstain":
            a += 1; bucket["abstain"] += 1
        elif is_correct(p, g):
            c += 1; bucket["correct"] += 1
        else:
            w += 1; bucket["wrong"] += 1
    n = len(ids)
    return {"n": n, "correct": c, "wrong": w, "abstain": a, "PA": (c - w) / n if n else 0.0,
            "accuracy": c / n if n else 0.0, "by_type": {k: dict(v) for k, v in by_type.items()}}
