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
    "hint_mismatch": "the engine function under ENGINE HINT gives a different value: {hint}{hint_note}. Use it (shape its result as "
                     "OUTPUT requires) unless the question has a condition it cannot express; if you keep your own calculation, print "
                     "one line saying why.",
    "format": "the value cannot be put in the required answer form ({kind}). Return exactly the form in the OUTPUT section.",
    # audits of the turn's code (answer/audit.py), raised only with DATAMATE_AUDIT=on, said once
    "entity_absent": "the question names {names}, but no string in the code is that value: filter with `== <value>` exactly as "
                     "spelled in RESOLVED INPUTS, not a substring, a code you typed or another name. If the code reaches the value "
                     "another way, keep the calculation and print one line saying how.",
    "profit_bound": "{bound} Check the cost side: unit cost x quantity in the same unit, only the period's purchases / expenses, "
                    "only valid orders.",
    "period_mismatch": "{mismatch} The Coder cannot change those periods: the spec must name the period the question asks for "
                       "(rulebook §2 dates).",
    "date_literal": "{stray} Filter with `period_start` / `period_end` (and `compare_start` / `compare_end`), do not type dates of "
                    "your own; if the question needs another period, say which one in a printed line.",
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
Return only JSON: {"verdict": "accept"|"revise"|"unanswerable", "target": "coder"|"explainer", "evidence": "<verbatim copy>",
"problem": "...", "fix": "..."}"""
# DATAMATE_JUDGE_FORMAT=v1: the earlier key order. `problem` and `fix` came before `evidence`, so a reply cut by the output cap lost the
# evidence first (dev300 2b4ca48e19: 8 of 287 replies cut, all read as "unreadable" = accept)
JUDGE_SYSTEM_V1 = JUDGE_SYSTEM.replace(
    '''"target": "coder"|"explainer", "evidence": "<verbatim copy>",
"problem": "...", "fix": "..."}''', '''"target": "coder"|"explainer", "problem": "...",
"fix": "...", "evidence": "<verbatim copy>"}''')
_FIELD = {k: re.compile(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"') for k in ("verdict", "target", "evidence", "problem", "fix")}


# a label the judge puts before a quote ("VERIFIED FACTS: ...", "ENGINE VALUE: ...", "Code Output: ...")
_LABEL = re.compile(r"^\s*(?:#\s*)?(?:verified facts|engine value|engine reference|engine hint|facts|output|printed output|code output|"
                    r"code|question|original question|spec)\b[^:]{0,20}:\s*", re.I)


def fact_fragments(evidence: str, facts: str) -> list[str]:
    """The pieces of a judge's evidence that are VERBATIM in the code-computed FACTS / ENGINE VALUE. The judge often joins two quotes
    with "..." or puts a label first ("VERIFIED FACTS: ... 19 ... ENGINE VALUE: item_qty(...) -> 32"): as a whole the evidence is
    in no text and the revise was dropped (dev300 q_c4eee7e4ca8d, q_40de33fe6cdc: right diagnoses). Only the FACTS count: a piece found
    in the question or the code alone is how format complaints about right answers got through when every piece was accepted."""
    if not evidence or not facts:
        return []
    # the counts and values only: the "# VERIFIED FACTS (...)" / "# ENGINE VALUE (...)" headers are boilerplate a revise could copy, and
    # so is a phrase every fact line has ("in the main period"): a piece counts only when it carries a number of the facts
    body = "\n".join(line for line in facts.splitlines() if not line.lstrip().startswith("#"))
    ff, numbers = fold_text(body), set(re.findall(r"\d+(?:[.,]\d+)*", body))
    out = []
    for part in re.split(r"\.\.\.|…|\bvs\.?\b", evidence):
        p = _LABEL.sub("", part).strip(" \t\n.;,'\"")
        if len(p.split()) >= 2 and fold_text(p) and fold_text(p) in ff and set(re.findall(r"\d+(?:[.,]\d+)*", p)) & numbers:
            out.append(p)
    return out


def _partial_json(text: str) -> dict | None:
    """The fields of a JSON reply cut before its end: each string value that is complete (its closing quote is there). None without a
    complete `verdict`. A revise keeps its evidence only when the whole quote arrived, so the verbatim filter still decides."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "{" not in text or "}" in text[text.rfind("{"):]:
        return None                                  # not cut: a JSON broken some other way is not read in pieces
    out = {}
    for k, rx in _FIELD.items():
        m = rx.search(text)
        if m:
            try:
                out[k] = json.loads(f'"{m.group(1)}"')
            except ValueError:
                out[k] = m.group(1)
    return out if out.get("verdict") in ("accept", "revise", "unanswerable") else None


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
    def __init__(self, model, gate: bool = True, judge_unanswerable: bool = False, judge_facts: bool = True,
                 judge_periods: bool = False, max_tokens: int = 1500, judge_format: str = "v2", evidence: str = "fragments"):
        self.model, self.gate, self.max_tokens = model, gate, max_tokens
        self.judge_format = judge_format       # v2: evidence before problem / fix, a cut reply read up to the cut; v1: as before
        self.evidence = evidence               # fragments: a revise whose evidence has pieces verbatim in the FACTS counts; strict: as before
        self.judge_unanswerable, self.judge_facts, self.judge_periods = judge_unanswerable, judge_facts, judge_periods

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
                                                  hint=getattr(getattr(st, "engine_hint", None), "text", ""),
                                                  hint_note=(f" ({st.engine_hint.note})" if getattr(getattr(st, "engine_hint", None), "note", "") else ""),
                                                  reshape=reshape_hint(st.answer_type, interp.kind),
                                                  names=", ".join(repr(x) for x in audit.get("entity_absent") or []),
                                                  bound=audit.get("profit_bound") or "", mismatch=audit.get("period_mismatch") or "",
                                                  stray=audit.get("date_literal") or "")
                      for i in issues]
            # the periods are the spec's: the Coder cannot fix a mismatch (the harness protects `period_*`), the Explainer can
            return Verdict("revise", "checks", "explainer" if "period_mismatch" in issues else "coder",
                           "\n".join(lines) + "\n" + _last_code(steps))
        if confirmed:
            return Verdict("accept", "engine_agrees", feedback=f"equals the exact engine function: {confirmed}"[:400])
        return self._judge(st, contract, steps, raw, interp)

    @staticmethod
    def periods_text(st) -> str:
        """The periods code computed, for the judge: the dates the question names (rulebook §2), the spec's periods, and the period
        issues the spec stage could not settle. "" when there is none of them."""
        named = [f"- {p.label}: {p.start} .. {p.end}" for p in getattr(st, "named_periods", None) or []]
        spec = [f"- {p.role} {p.label!r}: {p.start} .. {p.end}" for p in getattr(getattr(st, "period", None), "periods", None) or []]
        open_ = [i for i in getattr(getattr(st, "brief", None), "open_issues", None) or [] if str(i).startswith("periods:")]
        if not (named or spec or open_):
            return ""
        lines = ["# PERIODS (computed by code, rulebook dates; the Coder's `period_*` / `compare_*` must be these)"]
        lines += ["Named in the question:"] + (named or ["- (no calendar phrase)"])
        lines += ["Periods of the spec:"] + (spec or ["- (none)"])
        if open_:
            lines += ["NOT SETTLED by the spec stage:"] + [f"- {i}" for i in open_]
        return "\n".join(lines)

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
        facts = "\n\n".join(x for x in (self.facts_text(st) if self.judge_facts else "",
                                         self.periods_text(st) if self.judge_periods else "") if x)
        user = (f"ORIGINAL QUESTION: {st.question}\nSPEC (English rewrite): {spec}\nas_of_date = {st.as_of}\n\n{contract}\n\n"
                + (f"{facts}\n\n" if facts else "")
                + f"LAST CODE RUN AND ITS RESULT:\n{code}\n\n"
                + (f"THE CODE CONCLUDED UNANSWERABLE: {raw!r:.300}" if unanswerable
                   else f"ANSWER: {interp.kind} = {interp.raw!r:.300}"))
        v2 = self.judge_format != "v1"
        reply = chat(self.model, JUDGE_SYSTEM if v2 else JUDGE_SYSTEM_V1, user, stage="judge", max_tokens=self.max_tokens,
                     context=getattr(st, "context", None))
        v, cut = _json(reply), False
        if v is None and v2 and (v := _partial_json(reply)) is not None:
            cut = True                                   # the reply was cut: its complete fields only
        if v is None or v.get("verdict") not in ("accept", "revise", "unanswerable"):
            return Verdict("unanswerable" if unanswerable else "accept", "judge_ignored", feedback="(judge reply unreadable)")
        ev = str(v.get("evidence") or "").strip()
        target = v.get("target") if v.get("target") in ("coder", "explainer") else "coder"
        feedback = f"{v.get('problem', '')} -> {v.get('fix', '')}".strip(" ->") + (" [judge reply cut: read up to the cut]" if cut else "")
        if v["verdict"] == "revise":
            quote = fold_text(ev)
            found = next((name for name, text in (("question", st.question), ("code", code), ("facts", facts))
                          if quote and quote in fold_text(text)), "")
            if not found and self.evidence == "fragments" and fact_fragments(ev, facts):
                found = "facts_fragments"
            if not found:                                # no evidence: the checks in code passed, keep the answer
                return Verdict("unanswerable" if unanswerable else "accept", "judge_ignored", target, feedback, ev)
            return Verdict("revise", "judge", target, f"Judge: {feedback} (\"{ev}\")\n{code}", ev, found)
        if v["verdict"] == "unanswerable" and not unanswerable:
            # the judge thinks the data lacks the information, but the code computed a value: ask the coder to check
            return Verdict("revise", "judge", "coder", f"The judge suspects the data is insufficient: {feedback}. Check with "
                           "the data; if something is truly missing `answer = \"UNANSWERABLE: <reason>\"`, otherwise keep the calculation.", ev)
        return Verdict(v["verdict"], "judge", target, feedback, ev)
