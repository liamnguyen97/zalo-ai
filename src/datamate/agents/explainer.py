"""EXPLAINER (agents harness): an agent loop that turns the user's question into a SPEC clear enough to code from.

Input : question, as_of_date, rulebook, README, the Column Selector's schema, calendar facts around as_of_date.
Loop  : the model asks for LOOKUPS (find a value as stored in the data, describe a table/column, calendar facts) and
        writes a SPEC; the harness runs the lookups, checks the spec against the data, then a CLARITY CHECK call reads
        the original question next to the rewrite and lists what is still ambiguous or lost. Feedback goes back to the
        Explainer until the spec passes or the call budget ends.
Output: `Spec` (rewritten question, answer type and form, exact periods, exact entity spellings, tables/columns, the
        rulebook lines that decide the computation, answerability) and a numbered PLAN for the Coder.

No phrase lists: abbreviations, teencode, time expressions, entity names and the answer form are the model's reading,
checked against the DATA (values exist, dates valid, tables/columns exist) and by a second reading (clarity check).
Checks are feedback, never an abstain by themselves: an unfinished spec still goes to the Coder, marked as such
(the old Investigator abstained on every brief that failed validation: 79 of 300 on dev300b).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

from datamate.agents.lookups import DataTools
from datamate.agents.periods import fix_calendar_period, unmatched_periods
from datamate.context import Period, Resolved
from datamate.data.structure import resolve_table
from datamate.engines.reference import clean_call
from datamate.llm import chat
from datamate.text import fold_text

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")
MEASURES = ("quantity", "money", "orders", "count", "ratio", "name", "date", "yes_no", "other")
AMOUNTS = ("quantity", "money", "orders", "count")      # measures whose clash changes the number computed
TOOLS = ("find_values", "describe", "date_info", "shift_date", "lunar_to_solar")


@dataclass
class Spec:
    rewritten_question: str = ""
    answer_type: str | None = None
    measure: str | None = None                            # what the question measures (MEASURES)
    answer_form: str = ""                                 # e.g. "list of 2 numbers in order: new debt, repaid debt"
    k: int | None = None
    periods: list[dict] = field(default_factory=list)     # [{"label", "start", "end", "role": main|compare}]
    entities: list[dict] = field(default_factory=list)    # [{"value", "table", "column", "role"}]
    tables: dict = field(default_factory=dict)            # {table: [columns]}
    rules: list[str] = field(default_factory=list)        # rulebook / README lines that decide the computation
    answerable: bool = True
    unanswerable_reason: str = ""
    assumptions: list[str] = field(default_factory=list)
    engine_call: dict | None = None                       # {"function", "args"}: ONE exact engine call (engines/reference.py)
    missing_fields: list[str] = field(default_factory=list)   # properties the question needs that no column records (§7)
    unit: str | None = None                               # the unit the answer is asked in ("một thùng" -> "thùng"), fast mode
    rule_refs: list[str] = field(default_factory=list)    # rulebook row labels; code copies the lines into `rules` (fast mode)
    method: str = ""                                      # the plan, written in the spec call (DATAMATE_PLAN_MODE=spec)
    conditions: list[dict] = field(default_factory=list)  # [{"quote", "meaning"}]: what the question filters by, quoted (fast mode)
    status: str = "clear"                                 # clear | unchecked | fallback (for analysis)
    open_issues: list[str] = field(default_factory=list)  # what the checks still reported when the budget ended
    rule_sections: list[str] = field(default_factory=list)   # set by code: the rulebook sections of rule_refs (engines/routing.py)


CODE_FIELDS = ("status", "open_issues", "rule_sections")   # set by the harness, never taken from the model's JSON


LANG_RULE = """\
LANGUAGE: the questions, rulebook, README and data are VIETNAMESE; you think and write in ENGLISH. Free text you write
(rewritten_question, answer_form, assumptions, unanswerable_reason, plan) is English. But copy VERBATIM, never translate:
item / customer / group / staff names, column and table names, unit words (ly, đĩa, thùng, kg...), status labels, and any
`quote` (it must be a phrase that really appears in the Vietnamese question)."""

SYSTEM = """\
You are the EXPLAINER in a question-answering system over a shop's data. Your job: UNDERSTAND the question (it may use
abbreviations, no diacritics, teen-code, typos; it is Vietnamese) and rewrite it as a clear SPEC so a programmer can write
code that computes the answer WITHOUT guessing anything. Do NOT compute the answer.

""" + LANG_RULE + """

Each turn return ONE JSON:
  {"lookups": [{"tool": ..., "args": {...}}, ...], "spec": {...} or null}
Lookup tools (the system runs them and sends the results next turn):
  find_values {"text", "table"?, "column"?}  - the EXACT spelling in the data of a name the user typed (item, customer, group...)
  describe {"table", "column"?}              - the columns of a table, or all values / the range of one column
  date_info {"day"}                          - weekday, week (Mon-Sun), month, quarter, lunar date of a day
  shift_date {"day", "days"?, "weeks"?, "months"?, "years"?} - add / subtract time
  lunar_to_solar {"lunar_year", "lunar_month", "lunar_day"} - convert a lunar date to the solar calendar
Put ALL lookups in the first turn; look up only proper names (items, customers, groups...) and dates that need converting,
never ordinary words or abbreviations (understand those yourself: "dthu" = doanh thu = revenue, "hqua" = hôm qua = yesterday).
From the second turn on you must return a "spec".

spec fields:
  rewritten_question: the question rewritten in English, SELF-CONTAINED: names exactly as in the data (Vietnamese, verbatim),
      concrete dates YYYY-MM-DD, the measure and its unit, which orders count, the answer form; keep EVERY condition of the
      original question and add no new one;
  answer_type: number | percent | string | date | boolean | list | table
      (one name = string; several numbers/names in order = list; several rows and columns = table; "bao nhiêu"/"mấy" of one
       quantity = number; "có mấy/bao nhiêu mặt hàng|khách|đơn ..." COUNTS things = number, not a list of names;
       "tháng nào/ngày nào" = date);
  measure: what the question MEASURES, one of: quantity (amount of goods: ly, đĩa, phần, hộp, cái, kg, bao...) | money (revenue,
      debt, price, cost) | orders (number of orders) | count (number of distinct objects: how many items, how many customers)
      | ratio (percent, rate) | name | date | yes_no | other. The rewrite must measure EXACTLY this: "bao nhiêu ly/đĩa/phần" is
      quantity, NEVER turned into revenue;
  answer_form: a short description of the answer form (element order of a list, columns of a table, unit, rounding);
  k: number of elements for a top-k question, else null;
  periods: [{"label", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "role": "main"|"compare"}] - from the RULEBOOK and the date facts;
      "T3/2025", "th3", "T9" mean MONTH 3, month 9 (not a quarter; quarters are written "Q3", "quý 3");
      for a balance "up to the end of day X" (debt, stock) main runs from the first day of the data to X; no period: [];
  entities: [{"value": the EXACT value from find_values, "table", "column", "role": item|customer|group|staff|supplier|other}]
      - only things the QUESTION names, never the answer;
  tables: {"<table>": ["<column>", ...]} to use;
  rules: [the RULEBOOK / README lines that decide how to compute, copied verbatim (Vietnamese)];
  answerable: true/false; unanswerable_reason: why, when false (the data has no information the question needs). RULEBOOK §7:
    a question about a PROPERTY no column records (friendly, reputable, honest, enthusiastic) has answerable=false; do NOT
    substitute another quantity the data has (order count, revenue...) unless the question says to measure it that way;
  assumptions: [what you had to choose because the question is not explicit: one line per interpretation choice with a short
      reason; no editing history, no mention of earlier turns or the checker; names spelled exactly as in entities];
  engine_call: null, or {"function": "<name>", "args": {...}} when ONE engine function computes EXACTLY what the question asks
      with no extra condition, using the concrete dates / names of this spec. Functions: sales_metrics(start, end),
      item_qty(item, start, end), item_revenue(item, start, end), top_items(k, start, end, by='qty'|'amount'),
      item_price_on(item, day), item_qty_change(item, day, n), size_qty(size, start, end), debt_total(day),
      debt_by_customer(day, k), debt_customers(day), debt_aging(day), debt_repaid(start, end), debt_new(start, end),
      top_customers(k, start, end), purchase_total(start, end), purchase_avg_price(item, start, end, unit),
      top_supplier(start, end, k), expiring_items(day, n), expiring_value(day, n), nearest_expiry(item, day).
      Examples: {"function": "debt_repaid", "args": {"start": "2026-07-01", "end": "2026-07-28"}};
      {"function": "top_items", "args": {"k": 3, "start": "2026-03-01", "end": "2026-03-31", "by": "qty"}}.
      "best-selling" ranks by qty. Use null when no function fits exactly or you are unsure;
  missing_fields: [Vietnamese names of properties the question needs that NO column of the schema records, e.g. "giới tính
      khách" (customer gender) or "địa chỉ khách" (customer address) when no such column exists. RULEBOOK §7: information the
      shop does not record is unanswerable, not 0 and not guessed from another column. Empty list when every property the
      question needs has a column, or when you are not sure"""

CHECK_SYSTEM = """\
You are the SPEC CHECKER. Compare the ORIGINAL QUESTION (Vietnamese) with the SPEC (English rewrite + fields) and find REAL errors:
 1) Wrong / lost meaning: a condition dropped or added, wrong period or date, wrong item or customer, wrong k, wrong answer
    form, comparison periods in the wrong order.
 2) Still ambiguous: a programmer reading only the spec + rulebook + schema would still have to guess (a date not concrete,
    quantity / unit unclear, unclear what to count, unclear order of list elements).
How to check (1): list EACH condition / object / value in the rewrite (item, size, group, channel, payment method, customer,
status...) and find the matching phrase in the original question; a condition with no counterpart in the original (e.g. the
original asks only about size M but the rewrite adds size L; the original asks "how much money" but the rewrite adds "cash")
is an "added condition" error.
Default rules of the RULEBOOK apply by themselves and need not be restated. Ignore style. Do not compute the answer. Do not
guess conditions the question does not have.
Every error MUST carry evidence: "quote" is a phrase COPIED VERBATIM from the original Vietnamese question (or from the
rewrite) that shows the error; "field" is the spec field that is wrong. An error without evidence is dropped.
Return only JSON: {"clear": true|false, "issues": [{"field": "<field>", "quote": "<verbatim>", "problem": "...", "fix": "..."}]}"""

TYPE_SYSTEM = """\
Read the (Vietnamese) question and choose the ANSWER TYPE by RULEBOOK §9 (do not compute the answer):
  number | percent | string | date | boolean | list | table
  (ONE object, even "... nào nhất" = string, the top object; top k / listing several names = list;
   several numbers asked together = list in the order of the question; several rows and columns such as "k customers with
   their amounts" = table; "bao nhiêu"/"mấy" of one quantity = number, a percentage = percent; yes/no = boolean).
   "có mấy/bao nhiêu mặt hàng|khách|đơn" COUNTS = number; a question about a POINT IN TIME ("tháng nào", "ngày nào", even
   "... nào lớn nhất") = date, measure = date.
And the MEASURE the question measures: quantity (amount of goods: ly, đĩa, phần, hộp, cái...) | money (revenue, debt, price)
  | orders (number of orders) | count (number of distinct objects) | ratio (percent, rate) | name | date | yes_no | other.
"quote" must be COPIED VERBATIM from the Vietnamese question.
Return only JSON: {"answer_type": "...", "measure": "...", "quote": "<phrase of the question that decides the answer type>"}"""

PLAN_SYSTEM = """\
You are a data-analysis expert. From the checked SPEC write a SHORT PLAN in English (numbered, at most 8 steps) for computing
the answer with pandas: tables / columns and how to join them, the period filter (use exactly the dates in the spec), the
order-status filter by the rulebook, unit conversion, the calculation, the answer form. Keep names, columns and unit words
verbatim (Vietnamese). If an ENGINE FUNCTION computes exactly the asked quantity (with no extra condition) use it. REVENUE /
order count / average order: ALWAYS from `sales_metrics` (whole shop) or the `revenue` column of `sales_orders` (when there is a
condition: staff, channel, customer group, payment method, day, shift...), NEVER by adding up an order table's total column.
Purchases (nhập hàng): total money of slips = `purchase_total`; average import price of one item = `purchase_avg_price` (weighted by
quantity, in the asked unit); supplier ranking = `top_supplier`. NEVER divide a purchase slip's total column by an item's quantity:
it covers every item on the slip. Customer ranking: `top_customers`. Morning / afternoon shift: the `shift` column of `sales_orders` (values 'sáng' / 'chiều').
Item group / category, prescription drugs: columns of `item_lines` (already joined to the product table). Size: `size_qty`.
The ORIGINAL QUESTION is the authority: do not filter by a condition it does not state.
Do not write code."""

ENGINE_NOTE = """\
ENGINE FUNCTIONS available to the programmer (exact by the rulebook): sales_metrics(start, end) -> revenue/orders/avg_order for the
whole shop; item_qty / item_revenue(item, start, end); top_items(k, start, end, by='qty'|'amount'); item_price_on(item, day);
item_qty_change(item, day, n) -> % change of quantity over n days from day vs the n days before; size_qty(size, start, end);
debt_total(day, older_than_days=None); debt_by_customer(day, k); debt_customers(day); debt_aging(day, edges=(7, 15));
debt_repaid(start, end, customer=None); debt_new(start, end, customer=None); top_customers(k, start, end) (walk-in customers removed);
cancel_rate(start, end) = % of the orders placed in the period that were cancelled (denominator: every order, every status).
PURCHASES (when the shop has them): purchase_total(start, end) = money of the purchase slips dated in the period, cancelled slips left
out; purchase_avg_price(item, start, end, unit) = average import price of an item per `unit` ("một thùng" -> 'thùng'), weighted by the
quantity bought; top_supplier(start, end, k) -> [(supplier, money)]. A purchase slip's total column covers EVERY item on the slip.
LOTS (when the shop has them, §6): expiring_items(day, n) / expiring_value(day, n) = items / money of the stock left in lots expiring
in [day, day + n]; nearest_expiry(item, day) = the expiry date of the item's first-expiring lot still in stock.
TABLE `sales_orders`: one row per order, `revenue` already follows §1, `valid`, `customer`, `walk_in` + every column of the order
table: revenue / order count with a condition (channel, staff, customer group, payment, shift) = filter this table, then sum `revenue`."""


def date_facts(as_of: str, tools: DataTools) -> str:
    """Calendar facts around `as_of_date` the model computes periods from (generic, no phrase parsing)."""
    d = date.fromisoformat(as_of[:10])
    first = d.replace(day=1)
    prev_last = first - pd.Timedelta(days=1)
    lines = ["TODAY:\n" + tools.date_info(d),
             "YESTERDAY (hôm qua): " + tools.shift_date(d, days=-1),
             "LAST WEEK (Monday..Sunday): " + tools.shift_date(d, days=-d.weekday() - 7) + " .. " + tools.shift_date(d, days=-d.weekday() - 1),
             "LAST MONTH: " + prev_last.replace(day=1).isoformat() + " .. " + prev_last.isoformat()]
    cal = tools.calendar
    if not cal.empty and {"lunar_month", "lunar_day", "date"} <= set(cal.columns):
        tet = cal[(cal["lunar_month"] == 1) & (cal["lunar_day"] == 1)]
        lines.append("LUNAR NEW YEAR DAY 1 (Mùng 1 Tết): " + ", ".join(f"lunar year {int(r.lunar_year)} = {str(r.date)[:10]}" for r in tet.itertuples()))
    return "\n".join(lines)


def _json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


USAGE = {
    "find_values": '{"tool": "find_values", "args": {"text": "<name as the user typed it>"}}',
    "describe": '{"tool": "describe", "args": {"table": "<table>", "column": "<column, optional>"}}',
    "date_info": '{"tool": "date_info", "args": {"day": "YYYY-MM-DD"}}',
    "shift_date": '{"tool": "shift_date", "args": {"day": "YYYY-MM-DD", "days": -1}}',
    "lunar_to_solar": '{"tool": "lunar_to_solar", "args": {"lunar_year": 2025, "lunar_month": 12, "lunar_day": 23}}',
}


def run_lookups(tools: DataTools, lookups, limit: int = 8, done: set | None = None) -> list[str]:
    """Run the requested lookups. `done` remembers earlier calls: a repeat is not run again (a small model otherwise
    spends its whole budget asking the same thing)."""
    out = []
    done = set() if done is None else done
    for lk in (lookups if isinstance(lookups, list) else [])[:limit]:
        if not isinstance(lk, dict) or lk.get("tool") not in TOOLS:
            out.append(f"{lk!r}: unknown tool; tools: {list(TOOLS)}")
            continue
        args = lk.get("args") if isinstance(lk.get("args"), dict) else {}
        if not args:
            out.append(f"{lk['tool']}: missing args. Use this form: {USAGE[lk['tool']]}")
            continue
        key = json.dumps([lk["tool"], args], ensure_ascii=False, sort_keys=True)
        if key in done:
            out.append(f"{lk['tool']}({json.dumps(args, ensure_ascii=False)}): ALREADY LOOKED UP in an earlier turn (see the result above), not run again")
            continue
        done.add(key)
        fn = tools.functions()[lk["tool"]]
        try:
            res = fn(**args)
        except TypeError as e:   # wrong argument names
            res = f"error: {e}. Use this form: {USAGE[lk['tool']]}"
        except Exception as e:  # noqa: BLE001 - a bad call is feedback, not a crash
            res = f"error: {type(e).__name__}: {e}"
        out.append(f"{lk['tool']}({json.dumps(args, ensure_ascii=False)}) ->\n{res}")
    return out


AUTO_SCORE, AUTO_GAP = 1.0, 0.15   # DataTools.find: 1.4 = same text once accents/case are folded


def _datelike(v) -> bool:
    """A date, month or year written as a value ("2025-05-01", "tháng 5/2025", "05/2025"): it belongs in `periods`."""
    t = re.sub(r"(?i)\b(th[aá]ng|ng[aà]y|n[aă]m|qu[yý])\b", " ", str(v)).strip()
    if not re.search(r"\d", t) or re.search(r"[^\W\d_]{2,}", t):
        return False
    if re.fullmatch(r"\d{1,2}\s*/\s*\d{4}|\d{4}-\d{1,2}|\d{4}", t):
        return True
    try:
        pd.to_datetime(t, dayfirst=True)
        return True
    except (ValueError, OverflowError):
        return False


def _person_in_question(value: str, question: str) -> bool:
    """A person's name the question names: two of its words (or all, if shorter) are words of the question. Seen on
    holdout6: "bua qua cua hang co bn don" looked up as a customer and kept as 'Bạch Ngân Hằng' (only "hang" shared)."""
    words, q = fold_text(str(value)).split(), set(re.findall(r"\w+", fold_text(question)))
    return not words or sum(w in q for w in words) >= min(2, len(words))


ROLLING = re.compile(r"\b(\d{1,3}) ngay (?:gan nhat|gan day|vua qua|qua)\b")


def _fix_rolling_window(periods: list, question: str | None, as_of: str, notes: list) -> None:
    """Rulebook: "N ngày gần nhất" = N consecutive days INCLUDING today, the last day is today. The model wrote 31 days for
    "30 ngày gần nhất" (dev300 q_6f877dfe79e8: 321.9M instead of 313.9M), so the main period is set to exactly that."""
    m = ROLLING.search(fold_text(question or ""))
    main = next((p for p in periods if p.get("role") == "main"), None)
    if not m or main is None or int(m.group(1)) < 1:
        return
    end = date.fromisoformat(as_of[:10])
    start = end - pd.Timedelta(days=int(m.group(1)) - 1)
    if str(main.get("start"))[:10] != str(start) or str(main.get("end"))[:10] != str(end):
        notes.append(f"harness: period {main.get('label')!r} {main.get('start')}..{main.get('end')} -> {start}..{end} "
                     f"(the last {m.group(1)} days include today)")
        main["start"], main["end"] = str(start), str(end)


def _fix_month_to_date_compare(periods: list, question: str | None, notes: list) -> None:
    """Rulebook: month-to-date compared with the previous month uses the SAME NUMBER OF DAYS of the previous month (day 1
    to the day matching today's), not the whole month (dev300 q_657bc8d2b…: -42.34% instead of +8.99%)."""
    main = next((p for p in periods if p.get("role") == "main"), None)
    cmp_ = next((p for p in periods if p.get("role") == "compare"), None)
    if main is None or cmp_ is None or re.search(r"\bca thang\b|toan bo thang|nguyen thang", fold_text(question or "")):
        return
    try:
        a, b = date.fromisoformat(str(main["start"])[:10]), date.fromisoformat(str(main["end"])[:10])
        c, d = date.fromisoformat(str(cmp_["start"])[:10]), date.fromisoformat(str(cmp_["end"])[:10])
    except (KeyError, ValueError):
        return
    prev_last = a.replace(day=1) - pd.Timedelta(days=1)
    if not (a.day == 1 and (a.year, a.month) == (b.year, b.month)
            and (c, d) == (prev_last.replace(day=1), prev_last)) or b.day >= prev_last.day:
        return
    new_end = prev_last.replace(day=b.day)
    notes.append(f"harness: compare period {cmp_.get('label')!r} {c}..{d} -> {c}..{new_end} (same number of days as the month-to-date period)")
    cmp_["end"] = str(new_end)


WINDOW = re.compile(r"\b(\d{1,3}) ngay\b")


def _fix_adjacent_windows(periods: list, question: str | None, notes: list) -> None:
    """"N days from D vs the N days just before": both windows have N days. The model wrote a comparison window of 7 days
    for 14, 4 for 7 and 29 for 30 (dev300 eb8618, b06715, f3d30b), so when the comparison period ends the day before the
    main one and the question names N days, the main period is D..D+N-1 and the comparison N-1 days back from D-1."""
    main = next((p for p in periods if p.get("role") == "main"), None)
    cmp_ = next((p for p in periods if p.get("role") == "compare"), None)
    fq = fold_text(question or "")
    nums = {int(n) for n in WINDOW.findall(fq)}
    if main is None or cmp_ is None or len(nums) != 1 or not re.search(r"\btruoc\b", fq):
        return
    n = nums.pop()
    try:
        a, b = date.fromisoformat(str(main["start"])[:10]), date.fromisoformat(str(main["end"])[:10])
        c, d = date.fromisoformat(str(cmp_["start"])[:10]), date.fromisoformat(str(cmp_["end"])[:10])
    except (KeyError, ValueError):
        return
    if n < 2 or d != a - pd.Timedelta(days=1):
        return                                   # not "the N days just before the main period"
    new_main_end, new_c = a + pd.Timedelta(days=n - 1), a - pd.Timedelta(days=n)
    if (b, c) == (new_main_end, new_c):
        return
    notes.append(f"harness: periods {main.get('label')!r} -> {a}..{new_main_end} and {cmp_.get('label')!r} -> {new_c}..{d} "
                 f"(the question compares {n} days with the {n} days just before)")
    main["end"], cmp_["start"] = str(new_main_end), str(new_c)


def column_home(col: str, table: str, tables: dict, links) -> tuple[str, str] | None:
    """(table that has `col`, how it joins to `table`) for a column put on the wrong table: the table linked to `table`
    (parent / child, data/structure.py) when exactly one has it, else the only table that has it, else None."""
    having = [t for t, df in tables.items() if t != table and col in df.columns]
    if not having:
        return None
    joined = {}
    for l in links:
        if l.lines == table and l.head in having:
            joined[l.head] = f" (join `{table}`.`{l.line_key}` = `{l.head}`.`{l.key}`)"
        if l.head == table and l.lines in having:
            joined[l.lines] = f" (join `{l.lines}`.`{l.line_key}` = `{table}`.`{l.key}`)"
    if len(joined) == 1:
        return next(iter(joined.items()))
    return (having[0], "") if len(having) == 1 else None


def repair_spec(d, tables: dict, as_of: str, tools: DataTools | None = None,
                question: str | None = None) -> tuple[Spec | None, list[str], list[str], list[str]]:
    """Check a spec field by field against the data and keep everything that is right.

    Returns (spec, errors, warnings, notes). The spec is None only when `d` is not an object. A wrong field is dropped
    (and reported) instead of discarding the spec; an entity spelled differently from the data is replaced by the
    data's value when exactly one value matches clearly (notes say so). Errors go back to the model as feedback."""
    if not isinstance(d, dict):
        return None, ["spec must be a JSON object"], [], []
    s = Spec(**{k: d[k] for k in Spec.__dataclass_fields__ if k in d and k not in CODE_FIELDS})
    errs, warns, notes = [], [], []
    if not str(s.rewritten_question or "").strip():
        errs.append("rewritten_question is empty")
    if s.answer_type not in TYPES:
        errs.append(f"answer_type={s.answer_type!r} must be one of {list(TYPES)}")
        s.answer_type = None
    if s.measure is not None and s.measure not in MEASURES:
        errs.append(f"measure={s.measure!r} must be one of {list(MEASURES)}")
        s.measure = None
    if s.k is not None and not isinstance(s.k, int):
        try:
            s.k = int(s.k)
        except (TypeError, ValueError):
            errs.append(f"k={s.k!r} must be an integer or null")
            s.k = None
    periods = []
    for p in s.periods if isinstance(s.periods, list) else []:
        try:
            a, b = date.fromisoformat(str(p.get("start"))[:10]), date.fromisoformat(str(p.get("end"))[:10])
        except (ValueError, AttributeError):
            errs.append(f"period {p!r}: start/end must be YYYY-MM-DD (dropped)")
            continue
        if a > b:
            errs.append(f"period {p.get('label')!r}: start {a} is after end {b} (dropped)")
            continue
        if p.get("role") not in ("main", "compare"):
            errs.append(f"period {p.get('label')!r}: role must be 'main' or 'compare'")
        if str(b) > as_of[:10]:
            warns.append(f"period {p.get('label')!r} ends {b}, after as_of_date {as_of[:10]}: records after today have not "
                         "happened; keep it only if the question is truly about the future (expiry, due dates)")
        periods.append(p)
    if periods and not any(p.get("role") == "main" for p in periods):
        # "tháng nào cao nhất trong 3 tháng 10, 11, 12" lists equal periods, all read as 'compare'. As an error it held
        # the spec back from the check (public run: 6 month answers typed string, the date override never ran).
        periods[0] = {**periods[0], "role": "main"}
        notes.append(f"harness: period {periods[0].get('label')!r} -> role 'main' (no period had role 'main')")
    _fix_rolling_window(periods, question, as_of, notes)
    _fix_month_to_date_compare(periods, question, notes)
    _fix_adjacent_windows(periods, question, notes)
    fix_calendar_period(periods, question, as_of, notes)
    s.periods = periods
    ents = []
    for e in s.entities if isinstance(s.entities, list) else []:
        if not isinstance(e, dict):
            continue
        t, c, v = e.get("table"), e.get("column"), e.get("value")
        if t not in tables and isinstance(t, str) and (real := resolve_table(t, tables)) is not None:
            t, e = real, {**e, "table": real}           # the README's name for the table
        if question and e.get("role") in ("customer", "staff") and not _person_in_question(v, question):
            errs.append(f"entity {v!r} (role {e.get('role')}): the question does not name this person (dropped; the rewrite must not "
                        "name them either). List only entities the QUESTION names; ordinary words (hôm qua, cửa hàng...) are not customer names")
            if fold_text(str(v)) and fold_text(str(v)) in fold_text(str(s.rewritten_question or "")):
                # holdout6: the entity was dropped 4 turns running but the rewrite kept the name, and the Coder read it
                s.rewritten_question = question
            continue
        if t in tables and c in tables[t].columns and str(v) in set(tables[t][c].dropna().astype(str)):
            ents.append(e)
            continue
        if _datelike(v):
            errs.append(f"entity {v!r} is a date/month: dates belong in `periods`, not `entities` (dropped)")
            continue
        hits = []
        if tools is not None and v:
            ok_col = t in tables and c in tables[t].columns
            hits = tools.find(str(v), t if ok_col else None, c if ok_col else None, limit=4) or tools.find(str(v), limit=4)
        if hits and hits[0][0] >= AUTO_SCORE and (len(hits) == 1 or hits[0][0] - hits[1][0] >= AUTO_GAP):
            _, ht, hc, hv = hits[0]
            ents.append({**e, "value": hv, "table": ht, "column": hc})
            notes.append(f"harness: entity {v!r} -> value in the data {hv!r} ({ht}.{hc})")
            continue
        cands = ", ".join(f"{hv!r} ({ht}.{hc})" for _, ht, hc, hv in hits) or "none"
        errs.append(f"entity {v!r} is not in the data (dropped); closest candidates: {cands}. Pick a candidate or drop it "
                    "(from the rewrite too: it may be a typo of an ordinary word, e.g. 'ddax' = 'đã')")
        if question and fold_text(str(v)) and fold_text(str(v)) in fold_text(str(s.rewritten_question or "")):
            # holdout6 "xe ddax giao" (= "xe đã giao"): 'ddax' read as a plate, dropped 4 turns, kept in the rewrite
            s.rewritten_question = question
    s.entities = ents
    keep: dict[str, list] = {}
    links = getattr(getattr(tools, "structure", None), "links", None) or []
    for t, cols in (s.tables.items() if isinstance(s.tables, dict) else []):
        real = resolve_table(t, tables) if isinstance(t, str) else None
        if real is None:
            errs.append(f"no table {t!r} (dropped)")
            continue
        if real != t:
            notes.append(f"harness: table {t!r} -> {real!r} (the README's name for it)")
        t = real
        cols = [c for c in (cols if isinstance(cols, list) else []) if isinstance(c, str)]
        keep.setdefault(t, [])
        keep[t] += [c for c in cols if c in tables[t].columns and c not in keep[t]]
        missing = []
        for c in (c for c in cols if c not in tables[t].columns):
            home = column_home(c, t, tables, links)
            if home is None:
                missing.append(c)
                continue
            other, how = home
            keep.setdefault(other, [])
            if c not in keep[other]:
                keep[other].append(c)
            # biz_109 q_00a6e23cbacf: the slip date `Thời gian` was put on the slip LINES table; the spec stayed "unchecked"
            # on this error and the model never learned where the column was
            notes.append(f"harness: `{c}` is not a column of `{t}`; it is in `{other}`{how} -> moved there")
        if missing:
            errs.append(f"table {t!r} has no columns {missing} (dropped)")
    s.tables = keep
    s.engine_call = clean_call(s.engine_call) if s.engine_call else None
    s.missing_fields = [str(x) for x in s.missing_fields if str(x).strip()] if isinstance(s.missing_fields, list) else []
    if s.missing_fields:
        # a field the model says no column records, that a column of the shop IS named after: not missing (it would turn
        # `answerable` off, and with it the runner's coverage bounce)
        found = {f: c for f in s.missing_fields if (c := recorded_column(f, tables))}
        if found:
            s.missing_fields = [f for f in s.missing_fields if f not in found]
            notes.append("harness: missing_fields " + ", ".join(f"{f!r} is recorded (column `{c}`)" for f, c in found.items())
                         + " -> removed")
    if s.missing_fields and s.answerable is not False:
        s.answerable = False
        s.unanswerable_reason = "The data records no column for: " + ", ".join(s.missing_fields)
        notes.append(f"harness: marked possibly unanswerable, no column records {s.missing_fields} (the Coder must check with the data)")
    if s.answerable is False and not str(s.unanswerable_reason or "").strip():
        errs.append("answerable=false needs unanswerable_reason")
    return s, errs, warns, notes


def recorded_column(field_name: str, tables: dict) -> str | None:
    """`table`.`column` whose name is the field ("giới tính" -> `Giới tính`, `GioiTinh`, `Giới tính KH`): every word of the field, in
    order, is a word of the column name. None when no column is (or the field is too short to tell)."""
    from datamate.agents.coverage import _words      # the same CamelCase / underscore reading as the coverage check
    f = fold_text(field_name)
    if len(f.replace(" ", "")) < 4:
        return None
    pat = re.compile(rf"(?<!\w){re.escape(f)}(?!\w)")
    for t, df in (tables or {}).items():
        for c in df.columns:
            if c not in ("source_sheet", "status_std", "valid_order") and df[c].notna().any() and pat.search(_words(c)):
                return f"{t}`.`{c}"
    return None


def counts_names(measure: str | None, s: Spec) -> bool:
    """The readings count objects, the spec lists their names (no top-k): public run "có mấy mặt hàng nằm trong ít nhất
    122 đơn" answered with the names. "Which item" specs (k set, or a single name) are left alone."""
    return measure == "count" and s.measure == "name" and s.answer_type == "list" and not s.k


def _words_in(v: str, text: str) -> bool:
    fv = fold_text(str(v)).strip()
    return len(fv) >= 3 and re.search(rf"(?<!\w){re.escape(fv)}(?!\w)", fold_text(text)) is not None


def keep_anchored(s: Spec, anchored: dict, question: str) -> list[str]:
    """An entity the question names (word for word, folded) and the data has stays in every later spec.

    holdout6 r6 "nhom bia dong gop bnhiu phan tram doanh thu": the checker claimed 'bia' was not in the question,
    the Explainer dropped it and answered a customer-group question. Returns notes for the Explainer."""
    have = {fold_text(str(e.get("value"))) for e in s.entities}
    for e in s.entities:
        if _words_in(e.get("value"), question):
            anchored.setdefault(fold_text(str(e.get("value"))), e)
    notes = []
    for k, e in anchored.items():
        if k not in have:
            s.entities = list(s.entities) + [e]
            notes.append(f"harness: kept entity {e.get('value')!r} ({e.get('table')}.{e.get('column')}): the original question names it "
                         "and the data has it; it must not be dropped")
            if not _words_in(e.get("value"), str(s.rewritten_question or "")):
                s.rewritten_question = question
    return notes


def settled(err: str) -> str:
    m = re.match(r"entity '(.+?)' is not in the data", err)
    if m:
        return (f"harness: {m.group(1)!r} is not a value in the data (no item/customer/vehicle... like that): "
                "do NOT filter by it; it may be a typo of an ordinary word")
    return f"harness: a wrong field was dropped: {err[:160]}"


def spec_text(s: Spec) -> str:
    d = {k: v for k, v in asdict(s).items() if v not in ("", [], {}, None) and k != "status"}
    return json.dumps(d, ensure_ascii=False, indent=1)


def to_state(s: Spec, st) -> None:
    """Hand the spec to the later stages (Coder variables, Interpreter's expected type) through the shared state."""
    st.brief, st.answer_type = s, s.answer_type
    periods = []
    for p in s.periods:
        try:
            periods.append(Period(p.get("label") or "", date.fromisoformat(str(p["start"])[:10]),
                                  date.fromisoformat(str(p["end"])[:10]), p.get("role", "main")))
        except (ValueError, KeyError, TypeError):
            continue
    st.period = Resolved(date.fromisoformat(st.as_of[:10]), periods)


SYSTEM_FAST = """\
You are the EXPLAINER in a question-answering system over a shop's data. Your job: UNDERSTAND the question (Vietnamese; it may use
abbreviations, no diacritics, teen-code, typos) and write ONE SPEC so a programmer can compute the answer WITHOUT guessing.
Do NOT compute the answer.

""" + LANG_RULE + """

The system has already read the question in code (DATA MATCHES, PERIODS NAMED, ANSWER TYPE READ FROM THE WORDING): use those exact
spellings and dates unless the question clearly means something else; a reading marked "reliable" is right in ~98% of cases.
Return ONLY one JSON object, the spec, with these fields:
  rewritten_question: the question in English, SELF-CONTAINED (names verbatim from the data, dates YYYY-MM-DD, the measure and its unit,
      which orders count, the answer form); keep EVERY condition of the original question and add none;
  answer_type: number | percent | string | date | boolean | list | table
      (one name = string; several names or several numbers in order = list; rows with columns = table; "bao nhiêu"/"mấy" of one quantity
       = number; "có mấy mặt hàng|khách|đơn" COUNTS = number; "tháng nào/ngày nào" = date);
  measure: quantity (amount of goods) | money | orders | count (distinct objects) | ratio | name | date | yes_no | other.
      "bao nhiêu ly/đĩa/phần/thùng" is quantity, NEVER revenue;
  unit: the unit the answer is asked in ("một thùng" -> "thùng"; "bao nhiêu ly" -> "ly"), else null;
  answer_form: short description (element order of a list, columns of a table, unit, rounding);
  k: number of elements for a top-k question, else null;
  periods: [{"label", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "role": "main"|"compare"}]; for a balance "up to the end of day X"
      main runs from the first day of the data to X; no period: [];
  entities: [{"value": exact value from DATA MATCHES, "table", "column", "role": item|customer|group|staff|supplier|other}] - only what
      the QUESTION names, never the answer;
  tables: {"<table>": ["<column>", ...]} to use (exact names from the SCHEMA; a column lives on the table whose card lists it);
  rule_refs: the labels of the RULEBOOK rows / sections that decide the computation (e.g. "Giá nhập bình quân một mặt hàng trong
      tháng"), copied from the rulebook, labels only;
  engine_call: null or {"function": "<name>", "args": {...}} when ONE available function computes EXACTLY what is asked;
  answerable: true/false; unanswerable_reason: why, when false (RULEBOOK §7: a property no column records is unanswerable);
  missing_fields: [Vietnamese names of properties the question needs that NO column records], else [];
  assumptions: [only interpretation choices the question leaves open, one short line each; never a formula];"""

METHOD_FIELD = """
  method: the PLAN for the programmer, as numbered steps (as many as the computation needs): the engine function to call, or the tables,
      joins and filters (period, order status by the rulebook, conditions of the question only), unit conversion, the calculation, the
      answer form. Revenue / order counts come from `sales_metrics` or `sales_orders.revenue`, NEVER from an order total column; a
      document's total column covers EVERY item on it (never divide it by one item's quantity). Do not write code."""

CONDITIONS_FIELD = """
  conditions: [{"quote": "<words COPIED from the ORIGINAL QUESTION>", "meaning": "<the filter they set, in English>"}], one per condition
      the question states besides its period (item, group, active ingredient, channel, payment method, order status, customer, staff,
      size, shift...); [] when it states none. The rulebook's defaults are not conditions (they go in rule_refs). A condition you cannot
      quote from the question is NOT in the question: do not put it in the rewrite, entities, engine_call or method either."""

FIX_NOTE = ("\n\nYOUR PREVIOUS SPEC (JSON) AND WHAT THE CHECKS FOUND - return the FULL corrected spec, same fields:\n")


def _quoted(quote: str, question: str) -> bool:
    """The quote is words of the question: folded (accents may be added to a question typed without them) or both normalised (a
    quote may write "bn" / "cuwar" out as "bao nhiêu" / "cửa", as the written-out question does)."""
    from datamate.text import norm
    for a, b in ((fold_text(quote), fold_text(question)), (norm(quote), norm(question))):
        if a and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", b):
            return True
    return False


def check_conditions(s: Spec, question: str) -> list[str]:
    """Issues for the conditions that do not quote the question; those are dropped from the spec. The quote is checked in code, the
    way the judge's evidence and the clarity check's quotes are: dev300 q_83273acf61f7 "Doanh thu thuốc hoạt chất Vitamin C" got a
    rewrite about "prescription drugs", a §6 kê đơn rule and is_prescription in its engine_call, and the Coder answered 0 for 1,235,500;
    a "prescription" condition has no words of that question to quote."""
    kept, issues = [], []
    for c in s.conditions if isinstance(s.conditions, list) else []:
        if not isinstance(c, dict):
            issues.append(f"conditions: {str(c)[:80]!r} must be an object {{\"quote\", \"meaning\"}}")
            continue
        quote, meaning = str(c.get("quote") or "").strip(), str(c.get("meaning") or "").strip()
        if quote and _quoted(quote, question):
            kept.append({"quote": quote, "meaning": meaning})
            continue
        issues.append(f"conditions: {meaning or quote!r} quotes {quote!r}, which the question does not say: a condition the question "
                      "does not state must not be in the spec (rewrite, entities, engine_call, method); remove it, or quote the words "
                      "of the question that state it")
    s.conditions = kept
    return issues


def rule_sections(refs, rulebook: str) -> list[str]:
    """The rulebook sections (headings as written) the spec's `rule_refs` point to, read in code: by number ("§4. ..." -> "4. Công
    nợ"), else the section of the rulebook row whose label the ref contains ("Bán chạy: ..." -> "3. Xếp hạng"), else a heading whose
    title it contains. engines/routing.py reads them: a spec citing the debt or stock sections gets no sales hint."""
    from datamate.agents.guideline import parse_rows
    heads = [l.lstrip("#").strip() for l in (rulebook or "").splitlines() if l.startswith("#")]
    by_num = {m.group(1): h for h in heads if (m := re.match(r"(\d+)\.", h))}
    rows = [(fold_text(r.label), r.section) for r in parse_rows(rulebook or "")]
    titles = [(fold_text(re.sub(r"^\d+\.\s*", "", h)), h) for h in heads]
    out: list[str] = []
    for ref in refs if isinstance(refs, list) else []:
        f = fold_text(str(ref))
        m = re.match(r"\s*§?\s*(\d+)\b", str(ref))
        sec = by_num.get(m.group(1)) if m else None
        if sec is None:
            hits = [(len(lab), sec_) for lab, sec_ in rows if len(lab) >= 4 and lab in f]
            sec = max(hits)[1] if hits else None
        if sec is None:
            sec = next((h for t, h in titles if len(t) >= 4 and t in f), None)
        if sec and sec not in out:
            out.append(sec)
    return out


def rule_lines(refs, rulebook: str, limit: int = 6) -> list[str]:
    """The rulebook lines a spec refers to by label (a table row "| label | rule |" or a heading). Labels the rulebook does
    not contain are kept as written: the old `rules` field copied 300-670 characters of rulebook text per spec."""
    lines = [l.strip() for l in (rulebook or "").splitlines() if l.strip()]
    out = []
    for ref in refs if isinstance(refs, list) else []:
        f = fold_text(str(ref))
        if len(f) < 4:
            continue
        hit = next((l for l in lines if f in fold_text(l)), None)
        text = hit or str(ref)
        if text not in out:
            out.append(text)
    return out[:limit]


def code_checks(s: Spec, facts, question: str) -> tuple[list[str], list[str]]:
    """(notes: what code fixed, issues: what the model must fix). The model-free stand-ins for the blind type reading and the
    clarity check, each one tested on the training questions (agents/question.py)."""
    notes, issues = [], []
    r = facts.reading
    if r is not None and r.sure and s.answer_type != r.answer_type:
        if {s.answer_type, r.answer_type} == {"list", "table"}:
            issues.append(f"answer_type: the wording reads as {r.answer_type!r} ({r.rule}), the spec says {s.answer_type!r}: "
                          "check whether the question asks for names only (list) or names with amounts (table)")
        else:
            notes.append(f"harness: answer_type {s.answer_type} -> {r.answer_type} (question wording: {r.rule})")
            s.answer_type = r.answer_type
    if r is not None and r.sure and r.k and s.answer_type in ("list", "table") and s.k != r.k:
        # only a reading that is sure: "tháng 1/26 sp nào" read as k=26 under the default reading ruined a one-name answer
        notes.append(f"harness: k {s.k} -> {r.k} (question wording)")
        s.k = r.k
    if s.answer_type == "string" and s.k and s.k > 1:
        notes.append(f"harness: answer_type string -> list (k={s.k})")
        s.answer_type = "list"
    # a value the question names (several words: an item, group or customer, not a lone unit) and the spec lost
    used = fold_text(" ".join([str(e.get("value")) for e in s.entities] + [str(s.rewritten_question or ""), str(s.unit or "")]))
    for n in facts.names:
        if n.words >= 2 and fold_text(n.value) not in used and n.phrase not in used:
            t, c, v = n.hits[0]
            issues.append(f"entities: the question names {v!r} (`{t}`.`{c}`) but the spec does not use it")
    # an entity the question does not name (the model filled it from the data or invented it)
    qn = set(fold_text(question).split()) | set(" ".join(n.phrase for n in facts.names).split())
    for e in s.entities:
        w = fold_text(str(e.get("value"))).split()
        if w and sum(x in qn for x in w) < max(1, (len(w) + 1) // 2):
            issues.append(f"entities: {e.get('value')!r} is not named by the question; keep only what the question names")
    # a calendar phrase of the question with no period of the spec covering it. Covering = the same dates (or the union of several named
    # periods that follow each other): a spec quarter that merely contains the month the question names does not cover it. Where a spec
    # period may be wider than the phrase (a balance "đến hết tháng 5", "từ tháng 3 đến tháng 5", "7 ngày"), containing it is enough
    spans = [(str(x.get("start"))[:10], str(x.get("end"))[:10]) for x in s.periods]
    for p, strict in unmatched_periods(question, facts.periods, spans):
        issues.append(f"periods: the question names {p.label} ({p.start}..{p.end}) and no spec period " + ("equals it" if strict else "covers it"))
    return notes, issues


class ExplainerAgent:
    def __init__(self, model, check_model, plan_model, rulebook: str, calendar, max_calls: int = 5, max_checks: int = 2,
                 mode: str = "loop", plan_mode: str = "call", spec_max_tokens: int = 3000, written_out: bool = False,
                 conditions: bool = False):
        self.model, self.check_model, self.plan_model = model, check_model, plan_model
        self.rulebook, self.calendar = rulebook, calendar
        self.max_calls, self.max_checks = max_calls, max_checks
        self.mode, self.plan_mode, self.spec_max_tokens = mode, plan_mode, spec_max_tokens   # see run_fast
        self.written_out, self.conditions = written_out, conditions                           # settings.written_out / spec_conditions

    def _context(self, st, tools: DataTools) -> str:
        return (f"RULEBOOK:\n{self.rulebook}\n\nSHOP README:\n{st.ctx.readme[:6000]}\n\nSCHEMA:\n{st.schema}\n\n"
                f"DATE FACTS:\n{date_facts(st.as_of, tools)}\n\nas_of_date = {st.as_of}\nORIGINAL QUESTION: {st.question}")

    def blind_type(self, st, model=None) -> tuple[str | None, str, str | None]:
        """(answer type, quote, measure): an independent reading made WITHOUT seeing the spec. It counts only when its
        quote is really in the question: on holdout6 a reading quoted "tỉ lệ đơn huỷ" for "đếm số đơn đã huỷ" and
        talked the Explainer into a percent."""
        v = _json(chat(model or self.check_model, TYPE_SYSTEM, f"RULEBOOK:\n{self.rulebook}\n\nQUESTION: {st.question}", stage="blind_type", max_tokens=300))
        v = v if isinstance(v, dict) else {}
        t, m, q = v.get("answer_type"), v.get("measure"), str(v.get("quote") or "").strip()
        qw, words = re.findall(r"\w+", fold_text(q)), set(re.findall(r"\w+", fold_text(st.question)))
        if not qw or sum(w in words for w in qw) < 0.6 * len(qw):     # the reader may add accents or expand "bn"
            return None, q, None
        return (t if t in TYPES else None), q, (m if m in MEASURES else None)

    def clarity(self, st, s: Spec) -> tuple[bool, list[str], list[str]]:
        """(clear, issues kept, issues dropped). An issue is kept only when its quote is in the question (or the rewrite)
        and it names a real spec field: a small checker otherwise invents conditions the question never had."""
        user = (f"RULEBOOK:\n{self.rulebook}\n\nSCHEMA:\n{st.schema}\n\nas_of_date = {st.as_of}\n"
                f"ORIGINAL QUESTION: {st.question}\n\nSPEC:\n{spec_text(s)}")
        v = _json(chat(self.check_model, CHECK_SYSTEM, user, stage="clarity", max_tokens=1200))
        if v is None:
            return True, [], []          # a broken check must not block the answer
        where = fold_text(st.question) + " || " + fold_text(s.rewritten_question or "")
        kept, dropped = [], []
        for x in v.get("issues") or []:
            if not isinstance(x, dict):
                dropped.append(str(x)[:200])
                continue
            q, f = str(x.get("quote") or "").strip(), str(x.get("field") or "").strip()
            text = f"{f}: {x.get('problem', '')} -> {x.get('fix', '')} (\"{q}\")"
            (kept if q and fold_text(q) in where and f in Spec.__dataclass_fields__ else dropped).append(text)
        return not kept, kept, dropped

    def run(self, st, feedback: str = "") -> Spec:
        """Write the spec (and the plan). `feedback`: why the orchestrator asks for a new spec (an Evaluator verdict that
        the question was misread, with the previous spec), shown to the model with the question."""
        if self.mode == "fast":
            return self.run_fast(st, feedback)
        tools = DataTools(st.ctx.tables, self.calendar, getattr(st.ctx, "structure", None))
        base = self._context(st, tools)
        if feedback:
            base += "\n\nTHE PREVIOUS SPEC WAS JUDGED TO MISREAD THE QUESTION (write a new spec, fix exactly this):\n" + feedback
        history: list[str] = []
        log: list[dict] = []
        best, best_errs, done, checks, shown_warn, type_said = None, None, None, 0, False, False
        measure_said, measure_told, second = False, False, (None, "", None)
        last_issues: list[str] = []
        looked: set = set()
        blind = None
        seen_errs: set = set()
        anchored: dict = {}      # folded value -> entity the question names and the data has: never dropped later
        for i in range(self.max_calls):
            convo = "\n\n".join(history)[-12000:]
            must = ("\n\nLAST TURN: return only the spec." if i == self.max_calls - 1 else
                    "\n\nThis turn MUST contain a spec (lookups only if truly missing)." if i >= 1 else "")
            out = _json(chat(self.model, SYSTEM, base + ("\n\nPREVIOUS TURNS:\n" + convo if convo else "") + must, stage="explainer_turn"))
            entry = {"turn": i, "out": out}
            log.append(entry)
            if out is None:
                history.append(f"TURN {i}: (not JSON) -> return exactly ONE JSON")
                continue
            results = run_lookups(tools, out.get("lookups"), done=looked)
            feedback = list(results)
            if isinstance(out.get("spec"), dict):
                s, errs, warns, notes = repair_spec(out["spec"], st.ctx.tables, st.as_of, tools, st.question)
                notes += keep_anchored(s, anchored, st.question)
                # An error said once and repeated is settled: its field is already dropped. holdout6 r6 "xe ddax giao":
                # the same "'ddax' not in the data" error 5 turns running left the spec unchecked and the Coder
                # read 'ddax' as a licence plate.
                # Only "not in the data" entity errors: the field is gone and nothing else can be done. Other repeats
                # (no period with role 'main'...) still go back to the Explainer.
                repeated = [e for e in errs if e in seen_errs and re.match(r"entity '.+?' is not in the data", e)]
                seen_errs |= set(errs)
                errs = [e for e in errs if e not in repeated]
                if repeated:
                    s.assumptions = list(s.assumptions or []) + [settled(e) for e in repeated]
                entry.update(errors=errs, warnings=warns, notes=notes, repeated=repeated)
                if best is None or len(errs) <= len(best_errs):     # keep the most complete spec seen
                    best, best_errs = s, errs
                if notes:
                    feedback.append("THE SYSTEM FIXED:\n- " + "\n- ".join(notes))
                warn_turn = not errs and bool(warns) and not shown_warn
                if errs:
                    feedback.append("SPEC ERRORS (the wrong fields were dropped; fix them and return the full spec):\n- " + "\n- ".join(errs))
                    last_issues = errs
                elif warn_turn:
                    shown_warn = True
                    feedback.append("WARNING (fix it if right, keep it if the question truly needs it):\n- " + "\n- ".join(warns))
                # The checks run even when `repair_spec` dropped a field: the spec it returns is already repaired, and a
                # dropped column is no reason to leave the meaning unchecked. Before, any error skipped them: biz_109
                # q_00a6e23cbacf kept one dropped column as an error for the whole loop and a spec with a wrong formula
                # was never read by the type reader or the clarity check.
                # Only a dropped table / column the question does not need is settled: any other error (a bad period, an entity
                # not in the data...) means the spec will change, and a check spent on it is lost.
                benign = bool(errs) and all(re.match(r"(table '.+' has no columns|no table ')", e) for e in errs)
                checkable = bool(str(s.rewritten_question or "").strip()) and s.answer_type in TYPES
                if not warn_turn and checkable and (not errs or benign) and checks < self.max_checks:   # lookup results go back with the verdict
                    checks += 1
                    if blind is None:
                        blind = self.blind_type(st)
                        st.log["explainer_blind_type"] = {"answer_type": blind[0], "quote": blind[1], "measure": blind[2]}
                    issues = []
                    if blind[0] and blind[0] != s.answer_type and not type_said:
                        type_said = True       # said once: the Explainer may keep its choice with a reason
                        issues.append(f"answer_type: an independent reader chose {blind[0]!r} (\"{blind[1]}\"), the spec says "
                                      f"{s.answer_type!r}. See RULEBOOK §9; fix it if the reader is right, otherwise keep it and give the reason in assumptions")
                    if (blind[2] and blind[2] != s.measure and not measure_said
                            and blind[2] in AMOUNTS and (s.measure in AMOUNTS or counts_names(blind[2], s))):
                        # Only a clash between amounts (quantity / money / orders / count) changes the number computed;
                        # "date" vs "money" for "ngày nào doanh thu cao nhất" is the same question. One reading alone
                        # flipped 2 right specs on holdout6 r4 (plates -> revenue, pots -> orders): a second
                        # independent reading must agree before the Explainer hears it.
                        measure_said = True
                        second = self.blind_type(st, self.model)
                        st.log["explainer_blind_type2"] = {"answer_type": second[0], "quote": second[1], "measure": second[2]}
                    if measure_said and blind[2] != s.measure and second[2] == blind[2] and not measure_told:
                        measure_told = True
                        issues.append(f"measure: two independent readers both see the question measuring {blind[2]!r} (\"{blind[1]}\"), the spec says "
                                      f"{s.measure!r}. The rewrite must measure the original question's quantity; fix it if they are right, "
                                      f"otherwise keep it and give the reason in assumptions")
                    ok, kept, dropped = self.clarity(st, s)
                    issues += kept
                    entry["clarity"] = {"clear": not issues, "issues": issues, "dropped": dropped}
                    if not issues and (not errs or benign):
                        s.assumptions = list(s.assumptions or []) + [settled(e) for e in errs]
                        done = s
                        break
                    last_issues = errs + issues
                    if issues:
                        feedback.append("THE CHECKER STILL SEES PROBLEMS (fix them and return a new spec):\n- " + "\n- ".join(issues))
                elif not errs and not warn_turn:
                    done = s             # check budget used (or nothing to compare): take the last valid spec
                    done.status, done.open_issues = "unchecked", last_issues
                    break
            elif not results:
                feedback.append("Run lookups or return a spec.")
            history.append(f"TURN {i} YOU RETURNED:\n{json.dumps(out, ensure_ascii=False)[:3000]}\n\nRESULT:\n" + "\n\n".join(feedback))
        if done is None and (best is None or best_errs):
            # The last turn is the model's only chance to fix what is left, and a 9B model answers it with more lookups
            # (q_00a6e23cbacf: 5 turns, 3 of them lookups, the last one lookups again). One more call with the lookups
            # closed: it returns the spec or nothing; a spec counts when it has no more errors than the best one.
            forced = self._forced_spec(st, base, history, tools, anchored)
            if forced is not None:
                log.append({"turn": "forced", **forced["entry"]})
                if best is None or len(forced["errs"]) <= len(best_errs):
                    best, best_errs, last_issues = forced["spec"], forced["errs"], forced["errs"] or last_issues
        s = done or best
        if s is None:
            s = Spec(rewritten_question=st.question, status="fallback", open_issues=last_issues)
        elif done is None:
            s.status, s.open_issues = "unchecked", last_issues
        if not str(s.rewritten_question or "").strip():
            s.rewritten_question = st.question
        if blind is None:
            # the spec never reached the check (errors to the end, or no spec at all): the type / measure checks below
            # still need the independent reading. Public run: 45 of 621 had none; two "%" questions came out as number.
            blind = self.blind_type(st)
            st.log["explainer_blind_type"] = {"answer_type": blind[0], "quote": blind[1], "measure": blind[2], "late": True}
        if s.answer_type is None and blind and blind[0]:
            s.answer_type = blind[0]     # the spec never settled a type: take the independent reading
        if s.measure is None and blind and blind[2]:
            s.measure = blind[2]
        if (measure_told and blind[2] == second[2] and s.measure != blind[2]
                and (s.measure in AMOUNTS or counts_names(blind[2], s))):
            # two independent readings agree, were told to the Explainer, and it kept another amount with a reason:
            # holdout6 r6 "số đĩa ... bán được" (plates) kept as revenue. Across r3-r6 this held for 1 question, wrong.
            s.assumptions = list(s.assumptions or []) + [f"harness: measure {s.measure} -> {blind[2]} (two independent readers; "
                                                          "the original question is the authority, ignore rules/rewrites that say otherwise)"]
            s.measure, s.rewritten_question = blind[2], st.question
        if (blind and blind[2] and blind[2] == s.measure and s.answer_type in ("list", "string") and not s.k
                and (blind[0] in ("number", "percent") or s.measure == "count")):
            # both readings say the question measures one count / amount; the spec still asks for names
            new = blind[0] if blind[0] in ("number", "percent") else "number"
            s.assumptions = list(s.assumptions or []) + [f"harness: answer_type {s.answer_type} -> {new} (measure {s.measure})"]
            s.answer_type = new
        if s.answer_type == "string" and s.k and s.k > 1:
            # §9: one name is a string; several names are a JSON list, and "A, B" in one string is listed as WRONG (dev300
            # 019ec9: the Explainer read §9 as asking for a comma-separated string for a top-3 question)
            s.assumptions = list(s.assumptions or []) + [f"harness: answer_type string -> list (k={s.k}: several names are a list, not one string)"]
            s.answer_type = "list"
        if blind and "date" in (blind[0], blind[2]) and s.answer_type == "string":   # r5: type string, measure date
            # §9: a day or month answer is a date ('YYYY-MM-DD' / 'YYYY-MM'), never free text like "Tháng 11/2025"
            s.assumptions = list(s.assumptions or []) + ["harness: answer_type string -> date"]
            s.answer_type = "date"
        st.log["explainer"] = {"turns": log, "spec": asdict(s)}
        to_state(s, st)
        self.plan(st, s)
        return s

    # ------------------------------------------------------------------ fast mode (DATAMATE_P2=fast)
    def _fast_prompt(self, st, facts, tools) -> str:
        """Shared part first (rulebook, README, schema, functions: the same for every question of the shop, so the server's
        prefix cache serves it), the question's own part last."""
        have = ", ".join(sorted(k for k in getattr(st, "functions", []) or [])) or "(see ENGINE FUNCTIONS)"
        shop = ("" if getattr(st, "context", None) else      # shared: already at the start of the system prompt
                f"RULEBOOK:\n{self.rulebook}\n\nSHOP README:\n{st.ctx.readme[:6000]}\n\nSCHEMA:\n{st.schema}\n\n"
                f"{ENGINE_NOTE}\nAVAILABLE FUNCTIONS FOR THIS SHOP: {have}\n\n")
        return (shop + f"{facts.prompt()}\n\nDATE FACTS:\n{date_facts(st.as_of, tools)}\n\n"
                f"as_of_date = {st.as_of}\nORIGINAL QUESTION: {st.question}")

    def _spec_call(self, system: str, user: str, stage: str, context: str | None = None):
        out = _json(chat(self.model, system, user, stage=stage, max_tokens=self.spec_max_tokens, context=context))
        if isinstance(out, dict) and isinstance(out.get("spec"), dict):
            out = out["spec"]
        return out if isinstance(out, dict) else None

    def _settle(self, st, raw: dict, tools, facts, anchored: dict) -> tuple[Spec, list[str], list[str], list[str]]:
        """repair_spec + code checks: (spec, errors, issues for the model, notes)."""
        s, errs, warns, notes = repair_spec(raw, st.ctx.tables, st.as_of, tools, st.question)
        notes += keep_anchored(s, anchored, st.question)
        s.rules = rule_lines(s.rule_refs, self.rulebook) or [str(x) for x in (s.rules or [])]
        s.rule_sections = rule_sections(s.rule_refs, self.rulebook)
        cnotes, issues = code_checks(s, facts, st.question)
        if self.conditions:
            issues += check_conditions(s, st.question)
        else:
            s.conditions = []
        benign = [e for e in errs if re.match(r"(table '.+' has no columns|no table ')", e)]
        hard = [e for e in errs if e not in benign]
        return s, hard, issues, notes + cnotes + [settled(e) for e in benign] + [f"warning: {w}" for w in warns]

    def run_fast(self, st, feedback: str = "") -> Spec:
        """One spec call, checks in code, at most one fix call (agents/question.py reads the question first).

        Replaces the loop (turns asking for lookups, a blind type reading, one or two clarity checks, a plan call): on the first
        local runs the loop averaged 1.9 turns but every question still paid 6-8 model calls in this phase (69-74% of its time),
        and the checks changed nothing on 14 of 17 questions. Each check kept here runs in code; the model is called again only
        when they find something it must fix (an error repair_spec could not settle, a name the spec lost, an entity the
        question does not name, a period no spec period covers, a list / table disagreement)."""
        from datamate.agents.question import read_question
        tools = DataTools(st.ctx.tables, self.calendar, getattr(st.ctx, "structure", None))
        facts = read_question(st.question, st.as_of, st.ctx.tables, write_out=self.written_out)
        st.named_periods = list(facts.periods)
        system = SYSTEM_FAST + (CONDITIONS_FIELD if self.conditions else "") + (METHOD_FIELD if self.plan_mode == "spec" else "")
        user = self._fast_prompt(st, facts, tools)
        if feedback:
            user += "\n\nTHE PREVIOUS SPEC WAS JUDGED TO MISREAD THE QUESTION (write a new spec, fix exactly this):\n" + feedback
        log: dict = {"mode": "fast", "plan_mode": self.plan_mode, "facts": facts.as_log(), "calls": []}
        anchored: dict = {}
        ctx_text = getattr(st, "context", None)
        raw = self._spec_call(system, user, "spec", ctx_text)
        s, errs, issues, notes = (None, [], [], [])
        if raw is not None:
            s, errs, issues, notes = self._settle(st, raw, tools, facts, anchored)
        log["calls"].append({"call": "spec", "out": raw, "errors": errs, "issues": issues, "notes": notes})
        if issues and not errs and all(i.startswith("conditions:") for i in issues):
            log["fix_for_conditions_only"] = True        # what the quote check costs: a fix call no other check asked for
        if raw is None or errs or issues:
            fix = user + FIX_NOTE + (json.dumps(raw, ensure_ascii=False)[:4000] if raw else "(no JSON object was returned)")
            if notes:
                fix += "\nTHE SYSTEM FIXED:\n- " + "\n- ".join(notes)
            if errs or issues:
                fix += "\nTO FIX:\n- " + "\n- ".join(errs + issues)
            raw2 = self._spec_call(system, fix, "spec_fix", ctx_text)
            entry = {"call": "spec_fix", "out": raw2}
            if raw2 is not None:
                s2, errs2, issues2, notes2 = self._settle(st, raw2, tools, facts, anchored)
                entry.update(errors=errs2, issues=issues2, notes=notes2)
                if s is None or len(errs2) + len(issues2) <= len(errs) + len(issues):
                    s, errs, issues, notes = s2, errs2, issues2, notes2
            log["calls"].append(entry)
        if s is None:
            s = Spec(rewritten_question=st.question, status="fallback")
            r = facts.reading
            if r is not None:
                s.answer_type, s.k = r.answer_type, r.k
        elif errs or issues:
            s.status, s.open_issues = "unchecked", errs + issues
        if not str(s.rewritten_question or "").strip():
            s.rewritten_question = st.question
        if s.answer_type is None and facts.reading is not None:
            s.answer_type = facts.reading.answer_type
        s.assumptions = list(s.assumptions or []) + [n for n in notes if n.startswith("harness:")]
        log["spec"] = asdict(s)
        st.log["explainer"] = log
        to_state(s, st)
        if self.plan_mode == "spec":
            st.plan = s.method if isinstance(s.method, str) else "\n".join(f"{i}. {x}" for i, x in enumerate(s.method or [], 1))
            st.log["plan"] = st.plan
        elif self.plan_mode == "call":
            self.plan(st, s)
        else:
            st.plan = ""
        return s

    def _forced_spec(self, st, base: str, history: list[str], tools, anchored: dict) -> dict | None:
        """One extra call, lookups closed, asking for the full spec. None when the reply holds no spec."""
        convo = "\n\n".join(history)[-12000:]
        user = (base + ("\n\nPREVIOUS TURNS:\n" + convo if convo else "")
                + "\n\nFINAL TURN: lookups are closed. Return {\"lookups\": [], \"spec\": {...}} with the FULL spec now, "
                  "fixing every SPEC ERROR and THE SYSTEM FIXED note above. Do not ask for anything else.")
        out = _json(chat(self.model, SYSTEM, user, stage="explainer_forced"))
        if not isinstance(out, dict) or not isinstance(out.get("spec"), dict):
            return None
        s, errs, warns, notes = repair_spec(out["spec"], st.ctx.tables, st.as_of, tools, st.question)
        notes += keep_anchored(s, anchored, st.question)
        return {"spec": s, "errs": errs, "entry": {"out": out, "errors": errs, "warnings": warns, "notes": notes}}

    def plan(self, st, s: Spec) -> None:
        user = (f"RULEBOOK:\n{self.rulebook}\n\nSHOP README:\n{st.ctx.readme[:6000]}\n\nSCHEMA:\n{st.schema}\n\n"
                f"{ENGINE_NOTE}\n\nas_of_date = {st.as_of}\nORIGINAL QUESTION: {st.question}\nSPEC:\n{spec_text(s)}")
        st.plan = re.sub(r"<think>.*?</think>", "", chat(self.plan_model, PLAN_SYSTEM, user, stage="plan"), flags=re.S).strip()
        st.log["plan"] = st.plan
