"""CODER sub-agent: writes ONE block of pandas code per call from the task (context, checked spec, plan, output
contract), the blocks it already wrote this turn with what the RUNNER reported, and the feedback of the Evaluator /
orchestrator from earlier turns. The orchestrator alternates Coder and Runner until a block sets `answer` or the step
budget of the turn runs out."""
from __future__ import annotations

import re

from datamate.agents.guideline import render as render_guideline
from datamate.engines.reference import reference_note
from datamate.engines.routing import hint_note
from datamate.llm import chat


INSTRUCTIONS = """\
# ROLE
You are a data-analysis programmer. Write Python (pandas) code that COMPUTES the answer to a question about a shop's data,
following the SPEC (already checked against the data by the explainer) in the task. The shop's data, rulebook and README are
VIETNAMESE; your reasoning and the spec are English. Copy names, column names, status labels and unit words EXACTLY as they
appear in the data (Vietnamese, with diacritics). The task has: CONTEXT (rulebook, README, schema, available functions), TASK,
RESOLVED INPUTS, CALCULATION RULES, NOTES, PLAN and OUTPUT (the required answer form).

# SANDBOX VARIABLES
- `tables`: dict table name -> DataFrame, already cleaned (date columns are datetime, money is numeric). Use ONLY the table
  names listed in the schema, spelled exactly; `sales_orders` and `item_lines` (below) are also keys of `tables`; `calendar`; `as_of_date` ("today", 'YYYY-MM-DD'); `fold` (strip diacritics +
  lowercase, for comparing names).
- `period_start`, `period_end`, `compare_start`, `compare_end`: pd.Timestamp of the main / comparison period in RESOLVED INPUTS
  (`*_end` is the END of the last day, so that day is included; None when there is none). Filter dates with
  `(col >= period_start) & (col <= period_end)`. Never recompute these boundaries.
- `sales_orders` (ONE ROW PER ORDER, every status): `order`, `date` (order date), `valid` (False for cancelled AND returned
  orders: never use `~valid` to mean "cancelled"), `status_std` ('completed' | 'shipping' | 'cancelled' | 'returned'), `revenue`
  (the order's revenue by §1: sum of line amounts; 0 if not valid), `customer_key`, `customer` (name), `walk_in` (walk-in
  customer), `hour` / `shift` ('sáng' | 'chiều', by the order CREATION hour per the rulebook; use this column for shift
  questions) and EVERY column of the order table (channel, staff, payment, status, time...).
- `item_lines` (normalized sales lines, if present): order, date, valid, is_item, item, key, code, qty_base, base_unit, amount
  (`key == fold(name)`), PLUS every column of the product table such as group, category, prescription flag.
- dtype helpers (`pd.api.types` is blocked in this sandbox, use these names directly): `is_datetime64_any_dtype`,
  `is_numeric_dtype`, `is_string_dtype`, `is_object_dtype`, `is_integer_dtype`, `is_float_dtype`.

# ENGINE FUNCTIONS (exact by the rulebook; dates 'YYYY-MM-DD' or Timestamp, both ends included; usable for ANY period / item)
- `sales_metrics(start, end)` -> {'revenue','orders','avg_order'} for the whole shop (valid orders only, by order date).
- `find_items(text)`; `item_qty(item, start, end)` (base unit); `item_revenue(item, start, end)`; `item_base_unit(item)`;
  `top_items(k, start, end, by='qty'|'amount')` -> [(name, value)]; `item_price_on(item, day)`;
  `item_qty_change(item, day, n)` -> {'before','after','pct'} (n days from day vs the n days just before); `size_qty(size, start, end)`.
- Debt (§4): `debt_total(day, older_than_days=None)`, `debt_by_customer(day, k=None)` -> [(name, owed)], `debt_customers(day)`,
  `debt_aging(day, edges=(7, 15))` -> [0-7, 8-15, >15], `debt_repaid(start, end, customer=None)`, `debt_new(start, end, customer=None)`.
- Customers (§3): `top_customers(k, start, end)` -> [(customer name, revenue)], walk-in customers already removed.
- Cancel rate (§1, when listed): `cancel_rate(start, end)` -> % of the orders placed in the period that are cancelled, 2 decimals
  (the denominator is EVERY order of the period, every status).
- Purchases (§5, when listed under AVAILABLE FUNCTIONS): `purchase_total(start, end)` = money of the purchase slips dated in the period
  (cancelled left out); `purchase_avg_price(item, start, end, unit)` = average import price per `unit` (e.g. 'thùng'), weighted by the
  quantity bought; `top_supplier(start, end, k=1)` -> [(supplier, money)]. A slip's total column covers every item on the slip:
  never divide it by one item's quantity.
The functions of THIS shop are listed under "AVAILABLE FUNCTIONS" in the task: CALL them directly (do not test with
dir()/globals()); a function that is not listed means this shop has no data for it. Use a function when the task asks exactly
that quantity; a function cannot take extra conditions: then use `sales_orders` / `item_lines`.

# CALCULATION RULES
- Compute EXACTLY the quantity in OUTPUT: quantity = amount of goods CONVERTED TO THE BASE UNIT (a line "1 thùng" is 24 lon: use
  `item_qty` / `top_items` / `qty_base`, NEVER add the raw quantity column), NOT revenue; money = money; orders = number of orders
  (distinct order ids, `nunique`); count = number of distinct objects (ONE number, not a list of names).
- Revenue / order count / average order with a condition (channel, staff, customer group, payment method, shift, customer...):
  filter `sales_orders[sales_orders['valid'] & ...]`, then `revenue.sum()` / `order.nunique()`. NEVER add up the order table's
  total column (total after discount, order total...). Revenue of a group / category of goods = filter `item_lines` by that
  column, then sum `amount`.
- Properties of an item (prescription drug, group, category, active ingredient...): filter `item_lines` (already joined on the
  normalized name, so names that differ in case, diacritics or spacing still match). Do NOT `merge` the sales table with the
  product table by name yourself: names that do not match exactly lose their attributes and `fillna` misclassifies the line
  (biz_121: 82 lines, off by 7,000 đ).
- Order status: when a table has `valid_order` (True = completed / shipping), revenue / orders / best-seller use
  `df[df['valid_order']]`; cancelled / returned questions use `status_std == 'cancelled'` / `'returned'` (on `sales_orders` too:
  `~valid` counts cancelled AND returned orders together, 77 instead of 60 on dev300 1b04aa).
- Walk-in customers (no customer code) are NEVER the answer to a question about customers (§3).
- Item / customer names: use EXACTLY the values in RESOLVED INPUTS, compare with `==`.
- The ORIGINAL QUESTION is the authority: a condition / object in the rewrite that the original does NOT have (an added size,
  payment method, customer name...) must be IGNORED, do not filter by it.
- Rankings (top k, best seller, biggest customer): prefer `top_items` / `top_customers`; if you must rank yourself, break ties
  by code and keep the order the question asks for.
- §7: a question about a property NO column records (friendliness, reputation, honesty...) is UNANSWERABLE; do not measure it with
  order count / revenue unless the question says so. The task may say "may be unanswerable": that is only the schema-based
  opinion of the explainer, CHECK it with the data (print the date range, the relevant values) before concluding.

# RECIPES (rulebook definitions that are easy to get wrong; each reproduced the gold exactly)
- Customers who bought in a period = DISTINCT non-walk-in customers with at least one valid order (`valid & ~walk_in`, count
  `customer_key`). "Customers with at least N orders": count distinct valid orders per `customer_key`, keep >= N. A share is
  that count divided by the customers who bought.
- Cancelled orders: `status_std == 'cancelled'`; returned: `status_std == 'returned'`. `~valid` is both together.
- Shifts: the `shift` column of `sales_orders` ('sáng' 06:00-13:59, 'chiều' 14:00-21:59). Comparing afternoon with morning uses
  the MORNING as the base: (chiều - sáng) / sáng * 100.
- Average order value = revenue / number of valid orders in the same period (`sales_metrics(...)['avg_order']`), whole đồng.
- Nearest-expiry lot of a drug: among the lots still IN STOCK and NOT past expiry at today; never an expired or sold-out lot.
- Stock at the end of a day is the stock report, not the quantity sold (`item_qty` is sales).
- `item_qty` returns the item's BASE unit (`item_base_unit(item)`): if the question asks a bigger unit (thùng, hộp), convert with
  the README's conversion before answering.

# RESULT
- Assign the variable `answer` (do NOT call final_answer). `answer` must be COMPUTED from the data (a pasted number / string is refused).
- In the form given by OUTPUT: a plain number (a percent is the percent number: 12.4 = 12.4%), a name copied from the data,
  a date 'YYYY-MM-DD' (which month: 'YYYY-MM'), True/False, an ordered list (top k / listing names: ONLY the names, e.g.
  `[n for n, _ in top_items(3, s, e)]`, no numbers unless a table is asked), a table {"columns": [...], "rows": [[...], ...]}.
  Round as in §9 (money to whole đồng; percentages and non-integer averages to 2 decimals), never coarser.
- Not enough data: `answer = "UNANSWERABLE: <short reason>"`.

# HOW TO WORK
- Each turn reply with exactly ONE ```python ... ``` block. The system runs it in the SAME session (variables of earlier blocks
  stay) and sends back what it printed or the error. A block that assigns `answer = ...` ends the turn.
- Budget: at most 6 blocks. Write code that COMPUTES the answer from the FIRST block (with a few short `print`s to check).
  If a block failed, change the approach instead of resending the same code.
- Keep comments SHORT: at most one short line per step, no restating of the task or the rules (every word you write costs time).
- Do NOT re-assign `tables`, `calendar`, `period_*` / `compare_*` or the engine functions; do not create sample data.
- Parenthesize masks `df[(a) & (b)]`; compare datetime columns with pd.Timestamp; never print a whole table; no `__x__`
  attributes. Always start with `import pandas as pd`.
"""

# Per answer type: the exact form the submission needs (evaluation table, rulebook §9), told to the coder in OUTPUT.
TYPE_FORMAT = {
    "number": "ONE number (int/float), no unit; money and counts are integers, other numbers rounded to 2 decimals",
    "percent": "ONE percent number (12.4 means 12.4%), rounded to 2 decimals, with the right sign (+ increase / - decrease)",
    "string": "ONE name, copied VERBATIM from the data (with diacritics), no number or label",
    "date": "a string 'YYYY-MM-DD'; for a question about a MONTH use 'YYYY-MM' (not 'tháng 9/2025')",
    "boolean": "True or False",
    "list": "a Python list in EXACTLY the order the question gives (or the rulebook default); only the elements asked for",
    "table": '{"columns": [...], "rows": [[...], ...]} with exactly the columns asked for',
}
# Per measure: what to count, and the ready-made source that already applies the rulebook.
MEASURE_HINT = {
    "quantity": "amount of goods CONVERTED TO THE BASE UNIT (`item_qty` / `top_items(by='qty')` / `item_lines['qty_base']`), not money",
    "money": "an amount of money (revenue: `sales_metrics` / `sales_orders['revenue']` / `item_lines['amount']`)",
    "orders": "number of orders = number of distinct order ids (`sales_orders` filtered by `valid`, `order.nunique()`)",
    "count": "number of DISTINCT objects (one number, not a list of names)",
    "ratio": "a ratio / percent: numerator and denominator over the same period with the same filter rules",
}


# A list answer is names only or numbers only: 193 list answers in the training gold, none with a [name, amount] element. The
# spec's free-text `answer_form` said "each containing the item name and the total quantity sold" on 3 of 4 local runs of dev300
# q_33e8f7506387 (top 5); the judge enforced it and rejected the right names three times (abstain).
LIST_RULE = ("Rulebook §9: a list holds ONLY the values asked, the names (a top k, which items / customers) OR the numbers in the "
             "order asked; never [name, amount] pairs or dicts. This rule wins over any other description of the form.")
PAIR_FORM = re.compile(
    r"\b(?:pairs?|tuples?|dict(?:ionar(?:y|ies))?|key-value)\b|\beach (?:containing|with|having|including|showing)\b|"
    r"\b(?:names?|items?|products?|customers?|staff)\b[^.;]{0,40}\b(?:and|with|plus|along with|together with)\b[^.;]{0,30}"
    r"\b(?:quantit(?:y|ies)|qty|amounts?|revenue|values?|totals?|counts?|sales|money)\b", re.I)


def _period_lines(s, period) -> list[str]:
    """Each spec period with the sandbox variable that already holds it (only main + first compare have variables)."""
    out, has_var = [], {"main": "period_start .. period_end", "compare": "compare_start .. compare_end"}
    for p in s.periods:
        role = p.get("role", "main")
        var = has_var.pop(role, None)
        where = f"= variable `{var}`" if var else "(no variable of its own: filter with exactly these two dates yourself)"
        out.append(f"- {p.get('label') or role} [{'main period' if role == 'main' else 'comparison period'}]: "
                   f"{p.get('start')} .. {p.get('end')} (both ends included) {where}")
    return out


def coder_brief(s, rules: bool = True) -> str:
    """The Explainer's checked spec as the sections of the Coder's task (TASK .. OUTPUT). The spec's JSON stays for the
    Explainer's own check and plan; the Coder gets each part where it is used, the output contract last. `rules=False`: the
    spec's rulebook rows are shown in the GUIDELINE section instead (agents/guideline.py)."""
    model_notes = [a for a in s.assumptions or [] if not str(a).startswith("harness:")]
    harness_notes = [str(a).removeprefix("harness:").strip() for a in s.assumptions or [] if str(a).startswith("harness:")]
    ents = [f"- {e.get('role') or 'value'}: `{e.get('value')}` (table `{e.get('table')}`, column `{e.get('column')}`): filter with "
            f"`== {e.get('value')!r}`" for e in s.entities]
    tabs = [f"- `{t}`: " + ", ".join(f"`{c}`" for c in cols) for t, cols in (s.tables or {}).items()]
    parts = ["# TASK", f"Compute: {s.rewritten_question}"]
    parts += ["", "# RESOLVED INPUTS (checked against the data)", "Periods:", *(_period_lines(s, None) or ["- (the question names no period)"]),
              "Values to filter by (exact spelling in the data):", *(ents or ["- (none)"])]
    if tabs:
        parts += ["Tables / columns the explainer suggests (a hint; engine functions / `sales_orders` / `item_lines` still come first):", *tabs]
    if s.rules and rules:
        parts += ["", "# CALCULATION RULES (rulebook / README lines the explainer picked)", *[f"- {r}" for r in s.rules]]
    notes = []
    if not s.answerable:
        notes.append(f"The explainer thinks it may be UNANSWERABLE: {s.unanswerable_reason or '(no reason given)'}. "
                     "Check with the data before concluding.")
    if getattr(s, "status", "clear") != "clear":
        notes.append("The spec did NOT pass every check" + (": " + "; ".join(s.open_issues) if s.open_issues else "")
                     + ". The original question is the authority when they disagree.")
    notes += harness_notes
    if model_notes or notes:
        parts += ["", "# NOTES"]
        parts += [f"- System: {n}" for n in notes] + [f"- Explainer's assumption: {a}" for a in model_notes]
    fmt = TYPE_FORMAT.get(s.answer_type or "", "the form the question asks for")
    out = ["# OUTPUT (required)", f"- Answer type: {s.answer_type or '(unknown)'}: {fmt}."]
    if s.measure in MEASURE_HINT:
        out.append(f"- Measure: {s.measure} = {MEASURE_HINT[s.measure]}.")
    if s.k:
        out.append(f"- Number of elements: {s.k}.")
    if getattr(s, "unit", None):
        out.append(f"- Unit asked: {s.unit} (convert to it if the data records another unit).")
    form = s.answer_form
    if s.answer_type == "list" and form and PAIR_FORM.search(form):
        form = ""                      # "each containing the item name and the quantity": contradicts §9 (see LIST_RULE)
    if form:
        out.append(f"- Explainer's description: {form}")
    if s.answer_type == "list":
        out.append(f"- {LIST_RULE}")
    out.append('- Assign `answer = <expression computed from the data>`; not enough data: `answer = "UNANSWERABLE: <reason>"`.')
    return "\n".join(parts), "\n".join(out)


def engine_first(plan: str, ref) -> tuple[str, bool]:
    """The plan with a first step calling the engine reference, when the plan does not mention that function.

    dev5 biz_104 (top 5): the spec picked `top_items` but its `method` told the Coder to group and join by hand, and the Coder
    followed the method; b2: 3 of the 9 new abstains ranked customers by hand instead of calling `top_customers`. `ref` is the
    validated, trusted reference that already ran (engines/reference.py), never the raw `engine_call`."""
    if ref is None:
        return plan, False
    fn = ref.call["function"]
    if fn in (plan or ""):
        return plan, False
    args = ", ".join(f"{k}={v!r}" for k, v in ref.call["args"].items())
    step = (f"0. FIRST call `{fn}({args})`: the exact engine function for this question (its value is under ENGINE REFERENCE); "
            "shape its result as OUTPUT requires. The steps below matter only for a condition that function cannot express.")
    return step + ("\n" + plan if plan else ""), True


def build_task(st, rulebook: str, tools: list[str] | None = None) -> str:
    """The Coder's task, framed Context -> Task -> Inputs -> Rules -> Notes -> Plan -> Output.

    Stable text first (rulebook, README) so the server's prefix cache serves it across questions of a business; the
    question's own parts follow, and the output contract comes last, closest to where the model starts writing."""
    guideline = getattr(st, "guideline", None) or []
    body, output = coder_brief(st.brief, rules=not guideline)
    have = ", ".join(tools) if tools else "(none)"
    ref, hint = getattr(st, "engine_ref", None), getattr(st, "engine_hint", None)
    plan, _ = engine_first(st.plan, ref)
    context = ("" if getattr(st, "context", None) else       # shared: the rulebook / README / schema open the system prompt
               "# CONTEXT\n"
               f"## RULEBOOK\n{rulebook}\n\n## SHOP README\n{st.ctx.readme[:6000]}\n\n## AVAILABLE FUNCTIONS\n{have}\n\n"
               f"## SCHEMA (tables / columns picked beforehand)\n{st.schema}\n\n")
    return (context +
            f"## QUESTION\nas_of_date (today) = {st.as_of}\nORIGINAL QUESTION: {st.question}\n\n"
            f"{body}\n\n"
            + (f"# PLAN (suggested steps)\n{plan}\n\n" if plan else "")
            + (reference_note(ref) if ref is not None else "")
            + (hint_note(hint) if hint is not None else "")
            + (getattr(st, "facts", "") or "")                 # VERIFIED FACTS (agents/facts.py)
            + render_guideline(guideline)                      # the rulebook rows of this question, closest to OUTPUT
            + output)


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
OPEN_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*)\Z", re.S)


def extract_code(text: str) -> str | None:
    """The last fenced code block of the reply (a model sometimes shows an example first). A reply cut by the output cap
    has no closing fence: with no closed block, the code after the last ```python fence is used (the sandbox reports a
    cut-off line as a syntax error) instead of losing the block as "no code"."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    blocks = [b.strip() for b in CODE_BLOCK.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    m = re.search(r"<code>(.*?)</code>", text, re.S)
    if m:
        return m.group(1).strip()
    m = OPEN_BLOCK.search(text)
    return m.group(1).strip() or None if m else None


class Coder:
    def __init__(self, model):
        self.model = model

    def write(self, task: str, steps: list, feedback: str = "", temperature: float | None = None,
              context: str | None = None) -> str | None:
        """One code block, or None when the reply has none. `steps` are this turn's RunResults. `temperature` overrides
        the model's for this call (used to break out of a repeated block)."""
        parts = [task]
        if feedback:
            parts.append("# FEEDBACK FROM EARLIER TURNS (the previous answer was rejected: fix exactly this)\n" + feedback)
        if steps:
            parts.append("# CODE BLOCKS ALREADY RUN IN THIS TURN\n" + "\n\n".join(
                f"## Block {i}\n```python\n{r.code}\n```\nResult:\n{r.feedback()}" for i, r in enumerate(steps, 1)))
        parts.append("Write the NEXT code block (one ```python``` block).")
        return extract_code(chat(self.model, INSTRUCTIONS, "\n\n".join(parts), temperature=temperature, stage="coder", context=context))
