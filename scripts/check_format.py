"""Check answers against the official submission format (assets/evaluation-1.png).

usage: python scripts/check_format.py <run_dir | submission.jsonl> [--show N]
Reads the per-question logs (logs/*.json, field "answer") or a submission file and lists every line that breaks a rule.
"""
import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _two_dp(v) -> bool:
    return isinstance(v, int) or round(v, 2) == v


def problems(a: dict) -> list[str]:
    t, v = a.get("type"), a.get("answer")
    if t not in TYPES:
        return [f"unknown type {t!r}"]
    if t in ("unanswerable", "abstain"):
        return ["has an answer field"] if "answer" in a and v is not None else []
    if "answer" not in a or v is None:
        return ["missing answer"]
    if t in ("number", "percent"):
        if not _num(v):
            return [f"not a number: {type(v).__name__}"]
        return [] if _two_dp(v) else ["more than 2 decimals"]
    if t == "string":
        if not isinstance(v, str):
            return [f"not a string: {type(v).__name__}"]
        return ["empty string"] if not v.strip() else []
    if t == "date":
        if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v):
            return [f"not yyyy-mm-dd / yyyy-mm: {v!r}"[:80]]
        return []
    if t == "boolean":
        return [] if isinstance(v, bool) else [f"not true/false: {v!r}"[:60]]
    if t == "list":
        if not isinstance(v, list):
            return [f"not a list: {type(v).__name__}"]
        bad = [x for x in v if not (isinstance(x, str) or _num(x))]
        out = [f"list element not a name/number: {bad[0]!r}"[:80]] if bad else []
        if any(_num(x) and not _two_dp(x) for x in v):
            out.append("list number with more than 2 decimals")
        return out + (["empty list"] if not v else [])
    if t == "table":
        if not isinstance(v, dict) or set(v) != {"columns", "rows"}:
            return ["table must be {columns, rows}"]
        cols, rows = v["columns"], v["rows"]
        out = []
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            out.append("columns must be a list of names")
        if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
            return out + ["rows must be a list of lists"]
        if isinstance(cols, list) and any(len(r) != len(cols) for r in rows):
            out.append("row length != number of columns")
        cells = [c for r in rows for c in r]
        if any(not (isinstance(c, str) or _num(c)) for c in cells):
            out.append("cell not a name/number")
        if any(_num(c) and not _two_dp(c) for c in cells):
            out.append("cell number with more than 2 decimals")
        return out
    return []


def main() -> None:
    src = Path(sys.argv[1])
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 20
    if src.is_dir():
        rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(src / "logs" / "*.json")))]
        answers = [(r.get("id"), r.get("question", ""), r["answer"]) for r in rows]
    else:
        answers = [(a.get("id"), "", a) for a in map(json.loads, open(src, encoding="utf-8"))]
    types, bad = Counter(), []
    for i, q, a in answers:
        types[a.get("type")] += 1
        for p in problems(a):
            bad.append((p, i, q, a))
    print(f"{len(answers)} answers; types {dict(types)}")
    print(f"{len(bad)} format problems: {dict(Counter(p for p, *_ in bad))}")
    for p, i, q, a in bad[:show]:
        print(f"  {i} {p} | {q[:60]} | {json.dumps(a, ensure_ascii=False)[:160]}")


if __name__ == "__main__":
    main()
