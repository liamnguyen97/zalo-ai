"""COLUMN SELECTOR (agents harness): one model call that picks the tables and columns a question needs.

Input : the question, as_of_date, README, and the COLUMN DESCRIPTOR cards of every table (data/descriptor.py).
Output: `st.selection` = {table: [columns]} and `st.schema` = the cards of the picked tables first, then one line per
        other table. Nothing is hidden from the coder: `tables` still holds every table; the selection only decides what
        the Explainer and the Coder read first and in detail (dropping columns hurt in the past: the coder needed them).

The harness checks the answer against the data, not against wording: unknown tables / columns are reported back once
(with the closest real names); after that, whatever is valid is kept, and an empty pick means "all tables".
"""
from __future__ import annotations

import difflib
import json
import re

from datamate.data.structure import resolve_table
from datamate.llm import chat

SYSTEM = """\
You are the COLUMN SELECTOR of a question-answering system over a shop's data. Read the question (Vietnamese; it may use
abbreviations or no diacritics) and the table descriptions, and pick EVERY table and column needed to answer: measure
columns (money, quantity), date columns, order id, order status, customer, item, unit, group, and the keys that join tables.
Better too many than too few. Do not compute the answer. Table and column names must be copied exactly as written.
Return only ONE JSON: {"tables": {"<table name>": ["<column>", ...]}, "why": "<one sentence>"}"""


def _json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def check(pick, tables: dict) -> tuple[dict[str, list[str]], list[str]]:
    """(valid selection, problems). Problems name the closest real table / column."""
    out, errs = {}, []
    if not isinstance(pick, dict) or not isinstance(pick.get("tables"), dict):
        return out, ['answer must be {"tables": {table: [columns]}}']
    for t, cols in pick["tables"].items():
        if t not in tables:
            real = resolve_table(str(t), tables)      # the README's name for the table (sheet name without the file part)
            if real is None:
                close = difflib.get_close_matches(str(t), list(tables), n=3, cutoff=0.4)
                errs.append(f"no table {t!r}; closest: {close}")
                continue
            t = real
        have = [str(c) for c in tables[t].columns]
        keep = []
        for c in cols if isinstance(cols, list) else []:
            if c in have:
                keep.append(c)
            else:
                errs.append(f"no column {c!r} in `{t}`; closest: {difflib.get_close_matches(str(c), have, n=3, cutoff=0.4)}")
        out[t] = keep or have
    return out, errs


def focused_schema(descriptors: dict[str, str], tables: dict, selection: dict[str, list[str]]) -> str:
    """Cards of the picked tables (with the picked columns named), then a short line for every other table."""
    parts = []
    for t in selection:
        parts.append(descriptors.get(t, f"Table `{t}`") + "\n  SELECTED COLUMNS: " + ", ".join(f"`{c}`" for c in selection[t]))
    others = [f"`{t}` ({len(df)} rows): " + ", ".join(f"`{c}`" for c in df.columns) for t, df in tables.items() if t not in selection]
    if others:
        parts.append("OTHER TABLES (still usable if needed):\n- " + "\n- ".join(others))
    for k in ("__relations__", "__notes__"):
        if k in descriptors:
            parts.append(descriptors[k])
    return "\n\n".join(parts)


class ColumnSelector:
    def __init__(self, model, rounds: int = 1):
        self.model, self.rounds = model, rounds

    def run(self, st) -> None:
        tables, cards = st.ctx.tables, st.ctx.descriptors
        user = (f"SHOP README:\n{st.ctx.readme[:6000]}\n\nTABLES:\n{st.ctx.schema}\n\n"
                f"as_of_date = {st.as_of}\nQUESTION: {st.question}")
        pick = _json(chat(self.model, SYSTEM, user))
        selection, errs = check(pick, tables)
        log = [{"pick": pick, "errors": errs}]
        for _ in range(self.rounds):
            if not errs:
                break
            pick = _json(chat(self.model, SYSTEM, user + "\n\nPREVIOUS ANSWER:\n" + json.dumps(pick, ensure_ascii=False)
                              + "\nFIX THESE ERRORS (use the exact names in the table descriptions):\n- " + "\n- ".join(errs)))
            fixed, errs = check(pick, tables)
            selection = {**selection, **fixed}
            log.append({"pick": pick, "errors": errs})
        if not selection:  # nothing usable: every table, nothing lost
            selection = {t: [str(c) for c in df.columns] for t, df in tables.items()}
        st.selection = selection
        st.tables = tables
        st.schema = focused_schema(cards, tables, selection)
        st.log["column_selector"] = {"rounds": log, "selection": selection}
