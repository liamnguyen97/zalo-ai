"""EVALUATOR sub-agent: decides whether a turn's answer may be submitted.

Two layers:
1. Checks in code (no model call): no answer in the step budget, or the Interpreter's doubts (empty, wrong type,
   zero, walk-in customer ranked, placeholder value, table, day outside the period). Each comes with a concrete fix.
   On dev300b 16 of 18 values these checks rejected were wrong.
2. A judge (one model call) reads the question, the output contract, the code that produced the value and what it
   printed, and returns accept / revise / unanswerable. A revise must quote evidence found in the question, the code or
   the output; without it the judge is ignored (a small judge otherwise invents problems and turns right answers into
   abstains). A broken judge reply also accepts: the checks in code already passed. The judge checks only what code cannot
   (the quantity and the scope of the question); the answer's form is the Interpreter's. It also reads the VERIFIED FACTS
   (counts computed from the data) and the exact engine value, and may quote them as evidence: on the dev300 v3.5 logs at
   least 4 of the review's useful catches quoted facts or probe output (ledger #12).

No judge for an UNANSWERABLE the code concluded (it already survived the Runner's two bounces, for evidence and for data in
the period): on the dev300 v3.5 logs judging them hurt, 2 of the 3 questions the review damaged were a right UNANSWERABLE it
sent back, and submitting them directly takes PA 0.840 -> 0.857 on the same trajectories (`judge_unanswerable` brings the judge
back). Optionally no judge when the value equals an exact engine function's (`confirmed`, source `engine_agrees`); off by
default: on dev300 it saved 56 of 350 review calls (~0.4% of the time) and lost one question, an engine run on a misread period.

The verdict names who should fix it (`target`: coder | explainer); the orchestrator decides.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from datamate.llm import chat
from datamate.text import fold_text

ISSUE_TEXT = {
    "type": "answer type mismatch: the question needs `{expected}`, the code returned `{kind}`. If the computed values are right, "
            "keep the calculation and only change the form: {reshape}. Otherwise re-read the question: what value does it ask for?",
    "zero": "the result is 0. Usually a wrong filter (name spelled differently, status, date range, data type). Print the row "
            "count after EACH filter to find the step that loses the data; answer 0 only if the data really gives 0.",
    "sentinel": "the result is a placeholder value (999...9) in the data, not a real measurement. If the data only has the "
                "placeholder for this quantity, `answer = \"UNANSWERABLE: <reason>\"`.",
    "empty": "the result is empty / NaN. Print the row count after each filter to find the step that loses the data.",
    "walk_in": "the result contains a walk-in customer (khách lẻ / khách vãng lai): that is not a specific customer; remove them "
               "when ranking customers. With `sales_orders` filter `~walk_in` (already provided, no need to detect names); or use "
               "`top_customers`; with only the raw tables drop the names 'khách lẻ', 'kl', 'khách vãng lai' (compared after `fold`) "
               "and rank again.",
    "risky_type": "the table is malformed: it needs named `columns` and `rows` that are all as wide as the header "
                  "({{\"columns\": [...], \"rows\": [[...], ...]}}); or the question wants a single value / a list, not a table.",
    "table_rows": "a top-k table must have exactly k rows (k = {k}); fix the ranking or the cut-off.",
    "outside_period": "the returned date/month lies OUTSIDE the period the question asks about (period_start..period_end and the "
                      "comparison periods). Filter the data to the right period before looking for the largest day/month.",
    "engine_mismatch": "the exact engine function the explainer picked gives a different value: {ref}. Use the function (shape its "
                       "result as OUTPUT requires) unless the question has a condition it cannot express; if you keep your own "
                       "calculation, print one line saying why.",
    "format": "the value cannot be put in the required answer form ({kind}). Return exactly the form in the OUTPUT section.",
    # audits of the turn's code (answer/audit.py), raised only with DATAMATE_AUDIT=on, said once
    "entity_absent": "the question names {names}, but no string in the code is that value: filter with `== <value>` exactly as "
                     "spelled in RESOLVED INPUTS, not a substring, a code you typed or another name. If the code reaches the value "
                     "another way, keep the calculation and print one line saying how.",
    "profit_bound": "{bound} Check the cost side: unit cost x quantity in the same unit, only the period's purchases / expenses, "
                    "only valid orders.",
}

def reshape_hint(expected: str | None, kind: str | None) -> str:
    """How to turn the form the code returned into the one the question needs. dev5 biz_104: the top 5 came back with their
    quantities, the feedback only said "re-read the question", and the Coder rebuilt the ranking by hand and got it wrong."""
    one = {"number": "ONE number, no name and no unit", "percent": "ONE percent number (12.4 means 12.4%), no name"}
    if expected == "list" and kind == "table":
        return ("a flat Python list of ONLY the elements asked for: the names `[r[0] for r in rows]` (or `df.iloc[:, 0].tolist()`), "
                "or the numbers when the question asks for numbers")
    if expected in one and kind in ("list", "table"):
        return one[expected] + ": if you listed the objects to count them, `len(...)` of the distinct ones; if one value is asked, that value alone"
    if expected in one:
        return one[expected]
    if expected == "string":
        return "ONE name copied from the data, e.g. the first of a ranking `ranking[0][0]`, without its amount"
    if expected == "table":
        return '{"columns": [...], "rows": [[...], ...]} with exactly the columns the question asks for'
    if expected == "date":
        return "a string 'YYYY-MM-DD' (a month: 'YYYY-MM')"
    if expected == "boolean":
        return "True or False"
    return "the form in the OUTPUT section"


JUDGE_SYSTEM = """\
You are the ANSWER JUDGE of a question-answering system over a shop's data. Read the ORIGINAL QUESTION (Vietnamese), the SPEC
(English rewrite), the OUTPUT REQUIREMENT, the CODE that computed the value and what the code PRINTED, and decide whether the
code computes what the ORIGINAL question asks. Do not recompute. Check only:
(1) the QUANTITY: amount of goods vs money vs number of orders vs number of distinct objects, in the unit asked;
(2) the SCOPE: the period, the item / customer / group, and every condition of the ORIGINAL question (which orders count, channel,
staff, shift, ...): none dropped, none added (the original question wins over the spec).
Code already checks the rest, so NEVER revise for it: the answer's FORM (number / list / table shape, rounding; a list holds only
names or only numbers by rulebook §9, so never ask to add amounts, pairs or dicts to a list) and that the answer is the value the
code computed.
The VERIFIED FACTS and the ENGINE VALUE were computed by code from the data: you may quote them as evidence (for example a count
the answer cannot exceed, or an exact engine value the answer contradicts).
The code ran to completion; if the middle part is elided that is only how it is displayed, NOT an error: do not criticise the
code as "cut off" or "incomplete". Criticise only errors visible in the logic.
- accept: the answer fits the question.
- revise: a concrete error. "evidence" MUST copy VERBATIM a phrase from the question, the code or the printed output that shows
  the error; "target" = "coder" if the error is in the calculation, "explainer" if the question was misread (wrong period,
  object or quantity already in the spec).
- unanswerable: the data truly lacks the needed information (the code proved it: it printed the columns / date range).
Return only JSON: {"verdict": "accept"|"revise"|"unanswerable", "target": "coder"|"explainer", "problem": "...",
"fix": "...", "evidence": "<verbatim copy>"}"""


@dataclass
class Verdict:
    decision: str                 # accept | revise | unanswerable
    source: str                   # checks | no_answer | judge | judge_ignored | engine_agrees | unanswerable_direct | repeat_after_veto
    target: str = "coder"         # coder | explainer (who should fix a revise)
    feedback: str = ""            # what to fix, in words the Coder / Explainer reads
    evidence: str = ""
    found_in: str = ""            # where a judge revise's quote was found: question | code | facts (FACTS widen the filter: measured)

    def as_log(self) -> dict:
        return asdict(self)


def _json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _clip(code: str, limit: int) -> str:
    """Head and tail of long code with an explicit marker: the last lines (where `answer` is set) must stay visible."""
    if len(code) <= limit:
        return code
    head = limit * 3 // 5
    return code[:head] + "\n# ... (middle elided for brevity; the real code ran in full) ...\n" + code[-(limit - head):]


def _last_code(steps: list, limit: int = 1500) -> str:
    for r in reversed(steps):
        if r.code:
            return f"```python\n{_clip(r.code, limit)}\n```\n-> {r.feedback()[:800]}"
    return ""


class Evaluator:
    def __init__(self, model, gate: bool = True, judge_unanswerable: bool = False, judge_facts: bool = True):
        self.model, self.gate = model, gate
        self.judge_unanswerable, self.judge_facts = judge_unanswerable, judge_facts

    def evaluate(self, st, contract: str, steps: list, answered: bool, raw, interp, formatted: dict | None,
                 confirmed: str = "") -> Verdict:
        """`steps`: the turn's RunResults; `interp`: the Interpreter's result (None without an answer); `formatted`: the
        Formatter's line (None when the value could not be formatted); `confirmed`: the exact engine call whose value equals the
        answer (engines/reference.confirms), "" when there is none: then the checks in code are the whole review, no judge."""
        if not answered:
            errs = sorted({(r.error or "")[:200] for r in steps if r.error})
            stop = getattr(st, "stop_reason", "")
            return Verdict("revise", "no_answer", feedback=(
                f"The previous turn did not assign `answer` after {len(steps)} blocks (" + "; ".join(errs or ["no error, only exploration"])
                + ")." + (f" {stop[:1].upper() + stop[1:]}: do NOT send that code again, take a different route (an engine "
                          "function, `sales_orders` / `item_lines`, or print the row count after each filter)." if stop else "")
                + " Write fewer blocks: compute directly, print intermediate results, then assign `answer = <expression>`.\n"
                + _last_code(steps)))
        if interp.kind == "unanswerable":
            if not self.judge_unanswerable:
                return Verdict("unanswerable", "unanswerable_direct", feedback="the code concluded UNANSWERABLE after the "
                               "Runner's checks: submitted without the judge")
            return self._judge(st, contract, steps, raw, interp, unanswerable=True)
        issues = list(interp.issues) + ([] if formatted is not None else ["format"])
        if issues and self.gate:
            audit = getattr(st, "audit", None) or {}
            lines = [f"The previous turn returned {raw!r:.200} but a check rejected it:"]
            lines += ["- " + ISSUE_TEXT.get(i, i).format(expected=st.answer_type or "?", kind=interp.kind, k=getattr(st.brief, "k", None),
                                                  ref=getattr(getattr(st, "engine_ref", None), "text", ""),
                                                  reshape=reshape_hint(st.answer_type, interp.kind),
                                                  names=", ".join(repr(x) for x in audit.get("entity_absent") or []),
                                                  bound=audit.get("profit_bound") or "")
                      for i in issues]
            return Verdict("revise", "checks", feedback="\n".join(lines) + "\n" + _last_code(steps))
        if confirmed:
            return Verdict("accept", "engine_agrees", feedback=f"equals the exact engine function: {confirmed}"[:400])
        return self._judge(st, contract, steps, raw, interp)

    @staticmethod
    def facts_text(st) -> str:
        """What code computed from the data for this question: the VERIFIED FACTS counts and the engine value, if any."""
        parts = [(getattr(st, "facts", "") or "").strip()]
        for title, ref in (("ENGINE VALUE (the exact engine function the spec named)", getattr(st, "engine_ref", None)),
                           ("ENGINE VALUE (an engine function fitting the spec; advisory)", getattr(st, "engine_hint", None))):
            if ref is not None:
                parts.append(f"# {title}\n{ref.text}")
        return "\n\n".join(p for p in parts if p)

    def _judge(self, st, contract: str, steps: list, raw, interp, unanswerable: bool = False) -> Verdict:
        code = _last_code(steps, 6000)
        spec = getattr(st.brief, "rewritten_question", "") or "(none)"
        facts = self.facts_text(st) if self.judge_facts else ""
        user = (f"ORIGINAL QUESTION: {st.question}\nSPEC (English rewrite): {spec}\nas_of_date = {st.as_of}\n\n{contract}\n\n"
                + (f"{facts}\n\n" if facts else "")
                + f"LAST CODE RUN AND ITS RESULT:\n{code}\n\n"
                + (f"THE CODE CONCLUDED UNANSWERABLE: {raw!r:.300}" if unanswerable
                   else f"ANSWER: {interp.kind} = {interp.raw!r:.300}"))
        v = _json(chat(self.model, JUDGE_SYSTEM, user, stage="judge", max_tokens=800, context=getattr(st, "context", None)))
        if v is None or v.get("verdict") not in ("accept", "revise", "unanswerable"):
            return Verdict("unanswerable" if unanswerable else "accept", "judge_ignored", feedback="(judge reply unreadable)")
        ev = str(v.get("evidence") or "").strip()
        target = v.get("target") if v.get("target") in ("coder", "explainer") else "coder"
        feedback = f"{v.get('problem', '')} -> {v.get('fix', '')}".strip(" ->")
        if v["verdict"] == "revise":
            quote = fold_text(ev)
            found = next((name for name, text in (("question", st.question), ("code", code), ("facts", facts))
                          if quote and quote in fold_text(text)), "")
            if not found:                                # no evidence: the checks in code passed, keep the answer
                return Verdict("unanswerable" if unanswerable else "accept", "judge_ignored", target, feedback, ev)
            return Verdict("revise", "judge", target, f"Judge: {feedback} (\"{ev}\")\n{code}", ev, found)
        if v["verdict"] == "unanswerable" and not unanswerable:
            # the judge thinks the data lacks the information, but the code computed a value: ask the coder to check
            return Verdict("revise", "judge", "coder", f"The judge suspects the data is insufficient: {feedback}. Check with "
                           "the data; if something is truly missing `answer = \"UNANSWERABLE: <reason>\"`, otherwise keep the calculation.", ev)
        return Verdict(v["verdict"], "judge", target, feedback, ev)
