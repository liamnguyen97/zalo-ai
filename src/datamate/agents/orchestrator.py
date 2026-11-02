"""ORCHESTRATOR (main agent): holds the whole context of a question and runs the sub-agents in an agent loop.

  column_selector -> explainer -> [decide] -> turn 1..max_turns:
        coder <-> runner (until a block sets `answer` or `coder_steps` blocks)
        -> interpreter -> formatter -> evaluator --accept--> submit
                                                 --revise--> [decide]: coder (with feedback) | explainer | abstain
  after max_turns rejected turns: abstain (0 points beats a likely wrong answer, -1)

The orchestrator is a model at the points where there is a real choice ([decide]: after the spec, after a rejected
turn). Code is the guardrail: it offers only the actions allowed in the current state, falls back to the default when
the model picks anything else, caps Explainer re-runs and turns, and takes the single obvious step itself everywhere
else (no model call to say "now run the code"). Every sub-agent call is recorded in `log["orchestrator"]`.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from datamate.agents.coder import Coder, build_task, coder_brief, engine_first
from datamate.agents.facts import verified_facts
from datamate.agents.guideline import parse_rows
from datamate.agents.guideline import select as select_rules
from datamate.answer.evaluator import Evaluator
from datamate.answer.formatter import format_answer
from datamate.answer.interpreter import Interpretation, as_count, interpret
from datamate.engines.functions import TOOL_NAMES, rule_gated
from datamate.data.onboarding import summary
from datamate.agents.coverage import no_identified_customers, period_outside_data
from datamate.agents.shared import shared_context
from datamate.engines.reference import agrees, build_reference
from datamate.engines.routing import route
from datamate.llm import chat
from datamate.sandbox.runner import Runner, RunResult

REPEAT_TEMPERATURE = 0.5     # Coder temperature after it resubmitted a block it had already run (otherwise 0: reproducible)
STUCK_LIMIT = 2              # blocks that repeat a block or the previous error before the turn ends early


def _same_error(a: str, b: str) -> bool:
    """Two block errors that say the same thing (whitespace aside)."""
    return re.sub(r"\s+", " ", a or "").strip()[:400] == re.sub(r"\s+", " ", b or "").strip()[:400]

SYSTEM = """\
You are the ORCHESTRATOR (main agent) of a question-answering system over a shop's data. You see the question (Vietnamese),
the Explainer's spec and the results of earlier turns, and you decide the NEXT STEP. The assistants:
- coder: writes and runs code that computes the answer (the runner runs it, the interpreter + formatter put the value in the
  answer form, the evaluator judges it);
- explainer: rewrites the SPEC when the spec misreads the question (wrong period, object, quantity or answer form);
- abstain: give up the question (0 points) when no trustworthy answer is within reach; a wrong answer costs 1 point.
Pick coder when the fault is in the calculation; explainer when the spec itself disagrees with the original question; abstain
when the turns keep failing in the same way or the data is insufficient. Choose only from the ALLOWED ACTIONS. "message" is a
short, concrete instruction in English for the assistant you call (keep Vietnamese names verbatim).
Return only JSON: {"action": "...", "message": "...", "reason": "..."}"""


@dataclass
class Decision:
    action: str
    message: str = ""
    reason: str = ""
    by: str = "model"          # model | default (no valid model choice) | rule (only one action allowed)


def _json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def spec_summary(s) -> str:
    if s is None:
        return "(no spec yet)"
    periods = "; ".join(f"{p.get('label')}: {p.get('start')}..{p.get('end')} [{p.get('role')}]" for p in s.periods) or "(none)"
    ents = ", ".join(f"{e.get('value')!r}" for e in s.entities) or "(none)"
    lines = [f"- rewritten question: {s.rewritten_question}", f"- answer type: {s.answer_type}; measure: {s.measure}; k: {s.k}",
             f"- periods: {periods}", f"- entities: {ents}", f"- check status: {s.status}"]
    if s.open_issues:
        lines.append("- open issues: " + "; ".join(s.open_issues)[:600])
    if not s.answerable:
        lines.append(f"- the explainer thinks it is NOT answerable: {s.unanswerable_reason}")
    return "\n".join(lines)


class Orchestrator:
    def __init__(self, model, col_selector, explainer, coder: Coder, evaluator: Evaluator, inputs, rulebook: str,
                 max_turns: int = 3, coder_steps: int = 6, explainer_reruns: int = 1,
                 use_selector: bool = True, decide_after_explainer: bool = True, shared_prefix: bool = False,
                 guideline_rows: int = 4, verified_facts: bool = True):
        self.model, self.col_selector, self.explainer, self.coder, self.evaluator = model, col_selector, explainer, coder, evaluator
        self.inputs, self.rulebook = inputs, rulebook
        self.max_turns, self.coder_steps, self.explainer_reruns = max_turns, coder_steps, explainer_reruns
        # fast P2 (config.p2_mode): no Column Selector (it only re-ordered the cards: 94-109% of the full schema on small shops),
        # no model decision after the spec (3 of 17 local questions re-ran the Explainer there; on biz_124 it claimed
        # "August 2025 is in the future" of 2026-05-25 and turned a right engine value into a wrong "unanswerable")
        self.use_selector, self.decide_after_explainer = use_selector, decide_after_explainer
        self.shared_prefix = shared_prefix          # agents/shared.py: the shop's context opens every call
        self.gated = rule_gated(rulebook)           # engines whose rulebook row is not in this split's rulebook: off
        # Coder phase: the rulebook rows of the question before OUTPUT (agents/guideline.py), counts from the data (agents/facts.py)
        self.guideline_rows, self.verified_facts = guideline_rows, verified_facts
        self.rule_rows = parse_rows(rulebook) if guideline_rows else []

    # ------------------------------------------------------------------ bookkeeping
    def _call(self, st, agent: str, fn, **info):
        t0 = time.time()
        out = fn()
        st.log["orchestrator"].append({"agent": agent, "s": round(time.time() - t0, 2), **info})
        return out

    def decide(self, st, point: str, allowed: list[str], default: str, turns: list) -> Decision:
        """The orchestrator model's choice among `allowed`; the default when it gives no valid choice."""
        if len(allowed) == 1:
            d = Decision(allowed[0], by="rule")
        else:
            history = "\n".join(
                f"- Turn {t['turn']}: answer {json.dumps(t.get('submit') or t.get('raw'), ensure_ascii=False)[:200]}; "
                f"verdict: {t['verdict']['decision']} ({t['verdict']['source']}, fix at {t['verdict']['target']}): "
                f"{t['verdict']['feedback'][:400]}" for t in turns) or "(no turns yet)"
            user = (f"ORIGINAL QUESTION: {st.question}\nas_of_date = {st.as_of}\n\nSPEC:\n{spec_summary(st.brief)}\n\n"
                    f"TURNS SO FAR:\n{history}\n\nDECISION POINT: {point}\nALLOWED ACTIONS: {allowed} "
                    f"(default: {default})")
            v = _json(chat(self.model, SYSTEM, user, stage="orchestrator", max_tokens=500, context=getattr(st, "context", None)))
            if v and v.get("action") in allowed:
                d = Decision(v["action"], str(v.get("message") or "")[:600], str(v.get("reason") or "")[:300])
            else:
                d = Decision(default, by="default", reason=f"model reply not in {allowed}: {str(v)[:120]}")
        st.log["orchestrator"].append({"decide": point, "allowed": allowed, "action": d.action, "by": d.by,
                                       "message": d.message, "reason": d.reason})
        return d

    # ------------------------------------------------------------------ the loop
    def run(self, st) -> dict:
        st.log["orchestrator"], turns = [], []
        st.log["turns"] = turns
        if self.use_selector:
            self._call(st, "column_selector", lambda: self.col_selector.run(st))
        else:
            st.tables, st.selection, st.schema = st.ctx.tables, {}, st.ctx.schema
        try:   # the functions this shop has, named in the spec prompt (fast mode)
            st.functions = sorted(k for k, v in self.inputs.tools(st.ctx).items()
                                  if callable(v) and k in TOOL_NAMES and k not in self.gated)
        except Exception:  # noqa: BLE001
            st.functions = []
        st.context = (shared_context(self.rulebook, st.ctx.readme, st.schema, st.functions) if self.shared_prefix else None)
        self._call(st, "explainer", lambda: self.explainer.run(st))
        reruns = self.explainer_reruns
        if self.decide_after_explainer:
            d = self.decide(st, "after_explainer: use this spec or rewrite it?", ["coder"] + (["explainer"] if reruns else []),
                            "coder", turns)
        else:
            d = Decision("coder", by="rule", reason="fast P2: no decision after the spec")
            st.log["orchestrator"].append({"decide": "after_explainer", "allowed": ["coder"], "action": "coder", "by": "rule",
                                           "message": "", "reason": d.reason})
        feedback = ""
        if d.action == "explainer":
            reruns -= 1
            self._call(st, "explainer", lambda: self.explainer.run(st, d.message), rerun=True)
        elif d.message:
            feedback = f"Orchestrator: {d.message}"

        for turn in range(1, self.max_turns + 1):
            variables = self.inputs.variables(st)
            for name in self.gated:
                variables.pop(name, None)
            have = sorted(k for k, v in variables.items() if callable(v) and k in TOOL_NAMES)
            ob = getattr(st.ctx, "onboarding", None)
            if ob is not None:
                st.log["onboarding"] = summary(ob)
            # the reference is an "exact" number: not shown when the shop's engines failed their invariants (mis-detection)
            st.engine_ref = (build_reference(getattr(st.brief, "engine_call", None), variables)
                             if ob is None or ob.get("trusted") else None)
            # an advisory hint from the spec's shape when the Explainer named no call (it leaves it null on the standard shapes)
            st.engine_hint = (route(st.brief, variables, variables.get("period_start"), variables.get("period_end"), st.question)
                              if st.engine_ref is None and (ob is None or ob.get("trusted")) else None)
            if st.engine_hint is not None:
                st.log["orchestrator"].append({"engine_hint": st.engine_hint.text, "turn": turn})
            if st.engine_ref is not None:
                st.log["orchestrator"].append({"engine_reference": st.engine_ref.text, "turn": turn,
                                               "plan_engine_step": engine_first(st.plan, st.engine_ref)[1]})
            if gone := period_outside_data(st.brief, variables):                  # §7: no data is not 0
                st.log["orchestrator"].append({"coverage": gone, "turn": turn})
                return format_answer("unanswerable", "UNANSWERABLE: " + gone)
            cols = {str(c) for k in ("sales_orders", "item_lines") if variables.get(k) is not None for c in variables[k].columns}
            st.guideline = (select_rules(st.question, st.as_of, self.rule_rows, getattr(st.brief, "rules", None), have,
                                         variables.get("sales_orders"), getattr(st.brief, "measure", None), self.guideline_rows,
                                         columns=cols)
                            if self.guideline_rows else [])
            st.facts = verified_facts(st.question, st.brief, variables) if self.verified_facts else ""
            st.log.setdefault("coder_inputs", []).append({
                "turn": turn, "guideline": [f"{e.source}: {e.label or e.text[:60]}" for e in st.guideline],
                "facts": len(st.facts.splitlines()) - 1 if st.facts else 0})
            task = build_task(st, self.rulebook, have)
            _, contract = coder_brief(st.brief)
            steps = self._code_and_run(st, turn, task, variables, feedback)
            answered = bool(steps) and steps[-1].answered
            raw = steps[-1].value if answered else None
            interp = formatted = None
            if answered:                                                       # INTERPRETER
                if getattr(st.brief, "measure", None) == "count" and st.answer_type == "number":
                    raw = as_count(raw)
                try:
                    interp = self._call(st, "interpreter", lambda: interpret(raw, st.answer_type, st.brief), turn=turn)
                except Exception as e:  # noqa: BLE001 - an odd value rejects the turn (the Coder retries), never the question
                    st.log["orchestrator"].append({"interpreter_error": f"{type(e).__name__}: {e}"[:200], "turn": turn})
                    interp = Interpretation(st.answer_type or "string", raw, ["type"])
                if interp.issues == ["zero"] and any(t.get("issues") == ["zero"] for t in turns):
                    interp.issues = []     # a 0 sent back once and returned again after re-checking the filters is the data's
                if st.engine_hint is not None and interp.kind != "unanswerable":       # measured, not enforced
                    st.log.setdefault("routing", []).append({"turn": turn, "hint": st.engine_hint.text[:200],
                                                             "agrees": st.engine_hint.agrees_with(interp.raw)})
                if (st.engine_ref is not None and interp.kind != "unanswerable" and not agrees(st.engine_ref, interp.raw)
                        and not any("engine_mismatch" in t.get("issues", []) for t in turns)):
                    interp.issues = [*interp.issues, "engine_mismatch"]        # said once: a Coder that keeps its value is trusted
                if "walk_in" in interp.issues and (why := no_identified_customers(variables)):
                    st.log["orchestrator"].append({"coverage": why, "turn": turn})      # only walk-ins exist: nobody to rank
                    return format_answer("unanswerable", "UNANSWERABLE: " + why)
                try:                                                           # FORMATTER
                    formatted = self._call(st, "formatter", lambda: format_answer(interp.kind, interp.raw), turn=turn)
                except Exception:  # noqa: BLE001 - an unformattable value is the Evaluator's "format" issue
                    formatted = None
            verdict = self._call(st, "evaluator", lambda: self.evaluator.evaluate(st, contract, steps, answered, raw, interp,
                                                                                 formatted), turn=turn)
            turns.append({"turn": turn, "steps": [r.as_log() for r in steps], "raw": repr(raw)[:300] if answered else None,
                          "type": interp.kind if interp else None, "issues": interp.issues if interp else [],
                          "submit": formatted, "verdict": verdict.as_log()})
            if verdict.decision in ("accept", "unanswerable") and formatted is not None:
                return formatted
            if turn == self.max_turns:
                break
            allowed = ["coder", "abstain"] + (["explainer"] if reruns else [])
            default = verdict.target if verdict.target in allowed else "coder"
            d = self.decide(st, f"after_evaluator (turn {turn}/{self.max_turns} was rejected)", allowed, default, turns)
            if d.action == "abstain":
                break
            if d.action == "explainer":
                reruns -= 1
                prev = spec_summary(st.brief)
                self._call(st, "explainer", lambda: self.explainer.run(
                    st, f"{verdict.feedback}\n{d.message}\nPREVIOUS SPEC:\n{prev}"), rerun=True)
                feedback = ""                                                  # a new spec: the coder starts fresh
            else:
                note = f"Turn {turn}: {verdict.feedback}" + (f"\nOrchestrator: {d.message}" if d.message else "")
                feedback = (feedback + "\n\n" + note).strip()[-4000:]
        return {"type": "abstain"}

    def _code_and_run(self, st, turn: int, task: str, variables: dict, feedback: str) -> list[RunResult]:
        """CODER <-> RUNNER until a block sets `answer`, the step budget is used, or the Coder is stuck.

        Stuck = a block IDENTICAL to one that already ran, or a block that fails with the same error as the block before it
        (dev300 q_b445c50f6b5e: near-identical blocks kept failing the same way, 12 Coder calls, abstain; b2: 4 of the 9 new
        abstains came after a repeated block). After STUCK_LIMIT such blocks the turn ends early: the Evaluator's feedback
        starts the next turn instead of burning the rest of the budget."""
        runner, steps, seen, repeats, stuck = Runner(variables, getattr(st.brief, "answerable", False) is True), [], {}, 0, 0
        st.stop_reason = ""
        for _ in range(self.coder_steps):
            # a block that repeats an earlier one at temperature 0 can only fail the same way: the next call is sampled
            temp = REPEAT_TEMPERATURE if repeats else None
            code = self._call(st, "coder", lambda: self.coder.write(task, steps, feedback, temperature=temp, context=getattr(st, "context", None)), turn=turn,
                              **({"temperature": temp} if temp else {}))
            if not code:
                res = RunResult(code="", error="No code block found: reply with exactly one ```python ... ``` block")
            else:
                key = re.sub(r"\s+", " ", code).strip()
                if key in seen:
                    repeats, stuck = repeats + 1, stuck + 1
                    steps.append(RunResult(code=code, error=(
                        f"This block is IDENTICAL to block {seen[key]} and was not run again. Change the approach: print the "
                        "intermediate values, check the table and column names, or use `sales_orders` / `item_lines` / an engine function.")))
                    if stuck >= STUCK_LIMIT:
                        break
                    continue
                res = self._call(st, "runner", lambda: runner.run(code), turn=turn)
                if not res.refused:      # a refused block did not run: sent again, it may run (once-only guard checks)
                    seen[key] = len(steps) + 1
            prev = steps[-1] if steps else None
            steps.append(res)
            if res.answered:
                break
            if res.error and prev is not None and prev.error and _same_error(prev.error, res.error):
                repeats, stuck = repeats + 1, stuck + 1
                if stuck >= STUCK_LIMIT:
                    break
        if stuck >= STUCK_LIMIT and not (steps and steps[-1].answered):
            st.stop_reason = (f"the turn was ended after {len(steps)} blocks because the code kept failing the same way "
                              "(an identical block or the same error twice)")
            st.log["orchestrator"].append({"stuck": stuck, "blocks": len(steps), "turn": turn})
        return steps
