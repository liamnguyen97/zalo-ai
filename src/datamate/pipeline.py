"""The agentic pipeline: a main agent (agents/orchestrator.py) runs the sub-agents in an agent loop.

  sub-agents  COLUMN SELECTOR   agents/column_selector.py  picks the tables / columns to read first (1 model call)
              EXPLAINER         agents/explainer.py        question -> checked SPEC + plan (model loop with lookups)
              CODER             agents/coder.py            one block of pandas code per call
              RUNNER            sandbox/runner.py          runs it in the guarded sandbox, reports output / errors
              INTERPRETER       answer/interpreter.py      types the value against the spec, lists doubts
              FORMATTER         answer/formatter.py        the competition's exact answer format
              EVALUATOR         answer/evaluator.py        checks in code + a judge: accept | revise | unanswerable

Up to `max_turns` evaluated turns; after that many rejected turns the answer is `abstain` (0 points), never a guess
(-1). This module builds the agents once, caches each business's data, and turns failures into abstains.
"""
from __future__ import annotations

import threading
import time

from datamate.agents.coder import Coder
from datamate.agents.column_selector import ColumnSelector
from datamate.agents.explainer import ExplainerAgent
from datamate.agents.orchestrator import Orchestrator
from datamate.answer.evaluator import Evaluator
from datamate.config import Settings
from datamate.context import BusinessContext, QAState, Split
from datamate.llm import DeadlineExceeded, build_model, get_call_log, get_usage, reset_usage, set_deadline, set_session
from datamate.sandbox.variables import Runner as Inputs

__all__ = ["Pipeline", "Split"]


class Pipeline:
    run_end: float | None = None      # time.monotonic() the whole run must be over by (set_run_budget), None = no limit

    def __init__(self, split: Split, settings: Settings | None = None):
        self.split = split
        self.settings = s = settings or Settings()
        self.model = build_model(s)
        plan_model = build_model(s, thinking=True, max_tokens=s.thinking_max_tokens) if s.plan_thinking else self.model
        # the Explainer's independent type / measure reading must not be the same greedy call as its own reading
        check_model = build_model(s, temperature=s.check_temperature, seed=1001)
        self.orchestrator = Orchestrator(
            self.model, ColumnSelector(self.model),
            ExplainerAgent(self.model, check_model, plan_model, split.rulebook, split.calendar, s.explainer_calls,
                           s.explainer_checks, mode="fast" if s.p2_mode == "fast" else "loop",
                           plan_mode=s.plan_mode if s.p2_mode == "fast" else "call", spec_max_tokens=s.spec_max_tokens),
            Coder(self.model), Evaluator(self.model, s.gate, s.judge_unanswerable, s.judge_facts, s.judge_periods, s.judge_max_tokens,
                      s.judge_format, s.judge_evidence), Inputs(split.calendar),
            split.rulebook,
            s.max_turns, s.coder_steps, s.explainer_reruns,
            use_selector=s.p2_mode != "fast", decide_after_explainer=s.p2_mode != "fast", shared_prefix=s.shared_prefix,
            guideline_rows=s.guideline_rows, verified_facts=s.verified_facts, decide_mode=s.decide_mode,
            skip_judge_on_engine=s.skip_judge_on_engine, trust_repeat=s.trust_repeat, audit_mode=s.audit_mode,
            deadline_fallback=s.deadline_fallback, audit_skip=s.audit_skip,
            enforce_hints=s.enforce_hints)
        self._ctx: dict[str, BusinessContext] = {}
        self._ctx_lock = threading.Lock()

    def set_run_budget(self, seconds: float | None) -> None:
        """Start the run's clock: the questions that still wait when it runs out are abstained, the ones started near its end are quick."""
        self.run_end = time.monotonic() + seconds if seconds and seconds > 0 else None

    def budget(self) -> tuple[float | None, bool, bool]:
        """(seconds this question may take or None, the run is out of time, the run is nearly out of time) at this moment."""
        s = getattr(self, "settings", None) or Settings()
        left = None if self.run_end is None else self.run_end - time.monotonic()
        seconds = s.question_deadline_s if s.question_deadline_s and s.question_deadline_s > 0 else None
        if left is None:
            return seconds, False, False
        return (left if seconds is None else min(seconds, left)), left <= 0, left <= s.run_quick_s

    def business(self, business_id: str) -> BusinessContext:
        with self._ctx_lock:  # workers of the same business wait for one load instead of each parsing the files
            if business_id not in self._ctx:
                self._ctx[business_id] = BusinessContext.build(self.split.root / "data" / business_id)
            return self._ctx[business_id]

    def answer(self, q: dict) -> tuple[dict, dict]:
        """Return (submission_line, log). Never raises: failures become abstain."""
        t0 = time.time()
        log: dict = {"id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of_date": q["as_of_date"]}
        reset_usage()
        set_session(f"datamate-{self.split.root.name}-{q['business_id']}")
        seconds, run_over, quick = self.budget()
        log["budget_s"] = None if seconds is None else round(seconds, 1)
        set_deadline(None if seconds is None else time.monotonic() + seconds)
        try:
            if run_over:        # the run's time is gone: the question counts as abstain, as if it were missing from the file
                log["error"] = "DeadlineExceeded: the run's time budget is used up before this question started"
                line = {"id": q["id"], "type": "abstain"}
            else:
                st = QAState(q, self.business(q["business_id"]), log)
                st.quick = quick
                line = {"id": q["id"], **self.orchestrator.run(st)}
        except DeadlineExceeded as e:     # before the Orchestrator took over (loading the shop, ...)
            log["error"] = f"DeadlineExceeded: {e}"
            line = {"id": q["id"], "type": "abstain"}
        except (Exception, SystemExit) as e:  # noqa: BLE001 - SystemExit from generated code must not end the run
            log["error"] = f"{type(e).__name__}: {e}"
            line = {"id": q["id"], "type": "abstain"}
        finally:
            set_deadline(None)
        log["answer"] = line
        log["usage"] = get_usage()
        log["llm_calls"] = get_call_log()
        log["elapsed_s"] = round(time.time() - t0, 2)
        return line, log
