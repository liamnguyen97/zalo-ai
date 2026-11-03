"""Time budget (error-analysis P0): submission.jsonl complete after every question, a deadline per question and per run."""
import json
import time
from types import SimpleNamespace

import pytest

from datamate.cli import _infra_failure
from datamate.config import Settings
from datamate.eval.runs import SubmissionWriter
from datamate.llm import DeadlineExceeded, UsageModel, chat, check_deadline, set_deadline, time_left
from datamate.pipeline import Pipeline
from tests.test_orchestrator import RIGHT, ZERO, Scripted, _orch, _st


@pytest.fixture(autouse=True)
def _no_deadline_left_behind():
    yield
    set_deadline(None)


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ submission.jsonl
def test_submission_is_complete_after_every_question_and_in_question_order(tmp_path):
    path = tmp_path / "submission.jsonl"
    w = SubmissionWriter(path, ["q1", "q2", "q3"])
    w.add({"id": "q3", "type": "number", "answer": 3})            # finished first: the file already holds a whole, valid line
    assert _read(path) == [{"id": "q3", "type": "number", "answer": 3}]
    w.add({"id": "q1", "type": "string", "answer": "Phở bò"})
    assert [d["id"] for d in _read(path)] == ["q1", "q3"]          # question order, not finishing order
    assert not (tmp_path / "submission.jsonl.tmp").exists()        # swapped in atomically, nothing left beside it
    assert path.read_text(encoding="utf-8").count("Phở bò") == 1   # accents are written as they are
    w.add({"id": "q1", "type": "abstain"})                          # a question done again keeps ONE line
    assert [d for d in _read(path) if d["id"] == "q1"] == [{"id": "q1", "type": "abstain"}]
    assert len(w) == 2                                              # q2 is missing from the file: the scorer counts it as abstain


def test_a_resumed_run_starts_from_the_lines_it_already_has(tmp_path):
    path = tmp_path / "submission.jsonl"
    path.write_text('{"id": "q1", "type": "number", "answer": 1}\n{"id": "gone", "type": "abstain"}\nnot json\n', encoding="utf-8")
    w = SubmissionWriter(path, ["q1", "q2"], keep_existing=True)
    w.add({"id": "q2", "type": "number", "answer": 2})
    assert _read(path) == [{"id": "q1", "type": "number", "answer": 1}, {"id": "q2", "type": "number", "answer": 2}]
    fresh = SubmissionWriter(path, ["q1", "q2"])                    # a new run does not inherit anything
    fresh.add({"id": "q2", "type": "abstain"})
    assert _read(path) == [{"id": "q2", "type": "abstain"}]


# ------------------------------------------------------------------ llm.py
def test_chat_stops_before_a_call_that_starts_after_the_deadline():
    m = Scripted(judge=[{"verdict": "accept"}])
    set_deadline(time.monotonic() - 1)
    with pytest.raises(DeadlineExceeded):
        chat(m, "s", "u", stage="judge")
    assert m.seen == []                                             # no model call was made
    set_deadline(None)
    assert time_left() is None
    check_deadline()                                                # no deadline: nothing to raise


def test_every_request_is_cut_at_the_time_left(monkeypatch):
    seen = {}

    def fake(self, *a, **kw):
        seen.update(kw)
        return SimpleNamespace(content="{}", raw=None)
    monkeypatch.setattr("smolagents.OpenAIModel.generate", fake)
    model = object.__new__(UsageModel)
    set_deadline(time.monotonic() + 120)
    model.generate([])
    assert 100 < seen["timeout"] <= 120                             # not the 600 s of the client
    set_deadline(None)
    seen.clear()
    model.generate([])
    assert "timeout" not in seen                                    # no budget: the client's own timeout


# ------------------------------------------------------------------ orchestrator
def test_a_question_out_of_time_before_its_first_call_abstains_without_calling_the_model():
    m = Scripted(code=[RIGHT])
    st = _st()
    set_deadline(time.monotonic() - 1)
    assert _orch(m).run(st) == {"type": "abstain"}
    assert m.seen == [] and st.log["orchestrator"][-1]["submitted"] == "abstain"


class _TimeoutAfterJudge(Scripted):
    """The budget ends while the first turn is being judged."""

    def generate(self, messages, **kw):
        out = super().generate(messages, **kw)
        if self.seen[-1][0] == "judge":
            set_deadline(time.monotonic() - 1)
        return out


def test_the_latest_value_that_passed_the_checks_is_submitted_when_time_runs_out():
    judge = {"verdict": "revise", "target": "coder", "evidence": "bn cai", "problem": "p"}
    for fallback, expected in (("checked", {"type": "number", "answer": 2}), ("abstain", {"type": "abstain"})):
        m = _TimeoutAfterJudge(code=[RIGHT], judge=[judge])
        st = _st()
        set_deadline(time.monotonic() + 3600)
        assert _orch(m, deadline_fallback=fallback, trust_repeat=False).run(st) == expected
        assert st.log["turns"][0]["verdict"]["decision"] == "revise"
        assert "deadline" in st.log["orchestrator"][-1]
        set_deadline(None)


def test_a_value_that_failed_the_checks_is_never_the_fallback():
    m = _TimeoutAfterJudge(code=[ZERO])                             # 0 is rejected by the checks in code before any judge call
    st = _st()
    set_deadline(time.monotonic() + 3600)
    m.queues["orch"] = [{"action": "coder"}]
    # the deadline is passed after the first coder call: make the clock run out when the Coder is asked a second time
    calls = {"n": 0}
    orig = m.generate

    def gen(messages, **kw):
        out = orig(messages, **kw)
        calls["n"] += m.seen[-1][0] == "code"
        if calls["n"] == 1:
            set_deadline(time.monotonic() - 1)
        return out
    m.generate = gen
    assert _orch(m).run(st) == {"type": "abstain"}
    assert st.best is None


def test_a_quick_question_gets_one_turn_and_no_explainer_rerun():
    for quick, answers in ((False, {"type": "number", "answer": 2}), (True, {"type": "abstain"})):
        m = Scripted(code=[ZERO, RIGHT])
        st = _st()
        st.quick = quick
        assert _orch(m, max_turns=3).run(st) == answers
        assert len(st.log["turns"]) == (1 if quick else 2)


# ------------------------------------------------------------------ pipeline: the run's clock
def _pipe(**settings):
    p = object.__new__(Pipeline)
    p.settings = Settings(**settings)
    return p


def test_the_question_budget_is_the_smaller_of_its_own_and_what_the_run_has_left():
    p = _pipe(question_deadline_s=900, run_deadline_s=0, run_quick_s=1800)
    assert p.budget() == (900, False, False)                        # no run limit
    p.set_run_budget(10_000)
    assert p.budget()[0] == 900 and p.budget()[1:] == (False, False)
    p.set_run_budget(600)                                           # less than the question's own budget and than run_quick_s left
    seconds, over, quick = p.budget()
    assert 590 < seconds <= 600 and not over and quick
    p.run_end = time.monotonic() - 1
    assert p.budget()[1] is True
    assert _pipe(question_deadline_s=0).budget() == (None, False, False)    # 0 = no limit


def test_a_question_that_starts_after_the_run_is_out_of_time_is_abstained_and_redone_on_resume():
    p = _pipe(question_deadline_s=900)
    p.split = SimpleNamespace(root=SimpleNamespace(name="x"))
    p.business = lambda b: (_ for _ in ()).throw(AssertionError("the shop must not even be loaded"))
    p.run_end = time.monotonic() - 1
    line, log = p.answer({"id": "q1", "business_id": "b", "question": "q", "as_of_date": "2026-01-01"})
    assert line == {"id": "q1", "type": "abstain"} and _infra_failure(log)      # --resume-run answers it
    assert time_left() is None                                      # the thread's deadline is cleared after the question
