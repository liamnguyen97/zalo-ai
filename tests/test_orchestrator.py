"""Agentic loop: orchestrator decisions and guardrails, Coder <-> Runner, Evaluator (scripted model, real sandbox)."""
import json
from types import SimpleNamespace

import pandas as pd

from datamate.agents.coder import INSTRUCTIONS, Coder, extract_code
from datamate.agents.column_selector import ColumnSelector
from datamate.agents.explainer import (
    CHECK_SYSTEM,
    PLAN_SYSTEM,
    TYPE_SYSTEM,
    ExplainerAgent,
)
from datamate.agents.explainer import SYSTEM as EXPLAIN_SYSTEM
from datamate.agents.orchestrator import SYSTEM as ORCH_SYSTEM
from datamate.agents.orchestrator import Orchestrator
from datamate.answer.evaluator import JUDGE_SYSTEM, Evaluator
from datamate.answer.interpreter import Interpretation
from datamate.context import QAState
from datamate.sandbox.runner import Runner, RunResult
from datamate.sandbox.variables import Runner as Inputs

TABLES = {"don": pd.DataFrame({"Mã đơn": ["O1", "O2", "O3"], "Ngày": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]),
                               "Tên hàng": ["Bánh tiramisu", "Cà phê sữa (M)", "Bánh tiramisu"], "Thành tiền": [30.0, 25.0, 30.0]})}
SPEC = {"rewritten_question": "Tháng 5/2026 bán được bao nhiêu cái 'Bánh tiramisu'?", "answer_type": "number", "measure": "quantity",
        "periods": [{"label": "tháng 5/2026", "start": "2026-05-01", "end": "2026-05-31", "role": "main"}],
        "entities": [{"value": "Bánh tiramisu", "table": "don", "column": "Tên hàng", "role": "item"}],
        "tables": {"don": ["Tên hàng"]}, "rules": [], "answerable": True}
RIGHT = "```python\nd = tables['don']\nn = int((d['Tên hàng'] == 'Bánh tiramisu').sum())\nprint(n)\nanswer = n\n```"
ZERO = "```python\nd = tables['don']\nn = int((d['Tên hàng'] == 'tiramisu').sum())\nprint(n)\nanswer = n\n```"
LOOK = "```python\nprint(tables['don'].shape)\n```"
STAGES = {EXPLAIN_SYSTEM: "explain", TYPE_SYSTEM: "type", CHECK_SYSTEM: "check", PLAN_SYSTEM: "plan", INSTRUCTIONS: "code",
          JUDGE_SYSTEM: "judge", ORCH_SYSTEM: "orch"}


class Scripted:
    """Fake model: replies by stage (system prompt) from per-stage queues; the last reply repeats."""

    def __init__(self, **queues):
        self.queues, self.seen = {k: list(v) for k, v in queues.items()}, []
        self.queues.setdefault("select", [{"tables": {"don": ["Tên hàng", "Ngày"]}}])
        self.queues.setdefault("explain", [{"spec": SPEC}])
        self.queues.setdefault("type", [{"answer_type": "number", "measure": "quantity", "quote": "bn cai"}])
        self.queues.setdefault("check", [{"clear": True, "issues": []}])
        self.queues.setdefault("plan", ["1. đếm"])
        self.queues.setdefault("judge", [{"verdict": "accept"}])
        self.queues.setdefault("orch", [{"action": "coder", "message": ""}])

    def generate(self, messages, **kw):
        self.kwargs = getattr(self, "kwargs", []) + [kw]
        system, user = messages[0]["content"][0]["text"], messages[1]["content"][0]["text"]
        stage = STAGES.get(system, "select" if "COLUMN SELECTOR" in system else "?")
        self.seen.append((stage, user))
        q = self.queues.get(stage) or ["{}"]
        out = q.pop(0) if len(q) > 1 else q[0]
        return SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))

    def prompts(self, stage):
        return [u for s, u in self.seen if s == stage]


def _orch(m, **kw):
    return Orchestrator(m, ColumnSelector(m), ExplainerAgent(m, m, m, "RULEBOOK", pd.DataFrame()), Coder(m), Evaluator(m),
                        Inputs(pd.DataFrame()), "RULEBOOK", **kw)


def _st(question="tiramisu thang 5 ban dc bn cai"):
    ctx = SimpleNamespace(tables=TABLES, descriptors={"don": "Table `don`"}, readme="", schema="Table `don`", items=None,
                          sales=None, debt=None, orders=None)
    return QAState({"id": "q1", "question": question, "as_of_date": "2026-05-31"}, ctx, {})


# ------------------------------------------------------------------ Runner / Coder
def _runner():
    return Runner({"tables": {k: v.copy() for k, v in TABLES.items()}, "period_start": pd.Timestamp("2026-05-01")})


def test_runner_keeps_state_across_blocks_and_ends_on_answer():
    r = _runner()
    a = r.run("n = len(tables['don'])\nprint('rows', n)")
    assert a.output.strip() == "rows 3" and not a.answered and a.error is None
    b = r.run("answer = n * 2")
    assert b.answered and b.value == 6


def test_runner_reports_refused_code_and_asks_for_evidence_once():
    r = _runner()
    assert "tables" in r.run("tables = {}").error                          # protected input
    first = r.run("answer = 'UNANSWERABLE: không có cột'")
    assert not first.answered and "Before concluding UNANSWERABLE" in first.error
    assert r.run("answer = 'UNANSWERABLE: không có cột'").answered          # second time it stands


def test_extract_code_takes_the_last_block():
    assert extract_code("ví dụ:\n```python\nx = 1\n```\nthật:\n```python\nanswer = 2\n```") == "answer = 2"
    assert extract_code("<think>...</think>```py\nprint(1)\n```") == "print(1)"
    assert extract_code("không có code") is None
    assert extract_code("```python\nd = tables['don']\nanswer = len(d)") == "d = tables['don']\nanswer = len(d)"   # cut by the cap
    assert extract_code("```python\nx = 1\n```\nrồi:\n```python\nanswer = x +") == "x = 1"      # a closed block still wins
    assert extract_code("```python\n") is None


# ------------------------------------------------------------------ Evaluator
def _ev(judge):
    return Evaluator(Scripted(judge=[judge]))


def test_evaluator_checks_in_code_come_before_the_judge():
    st, steps = _st(), [RunResult(code="answer = 0", answered=True, value=0)]
    st.answer_type = "number"
    v = _ev({"verdict": "accept"}).evaluate(st, "C", steps, True, 0, Interpretation("number", 0, ["zero"]), {"type": "number", "answer": 0})
    assert v.decision == "revise" and v.source == "checks" and "the result is 0" in v.feedback
    v = _ev({"verdict": "accept"}).evaluate(st, "C", [RunResult(code="print(1)", output="1")], False, None, None, None)
    assert v.decision == "revise" and v.source == "no_answer"


def test_judge_needs_evidence_to_reject():
    st, steps = _st(), [RunResult(code="answer = n", output="2", answered=True, value=2)]
    it, line = Interpretation("number", 2), {"type": "number", "answer": 2}
    v = _ev({"verdict": "revise", "evidence": "khách VIP", "problem": "thiếu lọc"}).evaluate(st, "C", steps, True, 2, it, line)
    assert v.decision == "accept" and v.source == "judge_ignored"           # "khách VIP" is in neither question nor code
    v = _ev({"verdict": "revise", "evidence": "bn cai", "target": "explainer", "problem": "đếm cái"}).evaluate(st, "C", steps, True, 2, it, line)
    assert v.decision == "revise" and v.target == "explainer"
    v = _ev({"verdict": "unanswerable", "problem": "thiếu cột"}).evaluate(st, "C", steps, True, 2, it, line)
    assert v.decision == "revise" and "UNANSWERABLE" in v.feedback          # a computed value is not dropped silently
    assert _ev("not json").evaluate(st, "C", steps, True, 2, it, line).decision == "accept"


# ------------------------------------------------------------------ Orchestrator loop
def test_first_turn_accepted():
    m = Scripted(code=[RIGHT])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    agents = [e.get("agent") or "decide" for e in st.log["orchestrator"]]
    assert agents == ["column_selector", "explainer", "decide", "coder", "runner", "interpreter", "formatter", "evaluator"]
    assert len(st.log["turns"]) == 1 and st.log["turns"][0]["verdict"]["decision"] == "accept"


def test_a_rejected_turn_returns_to_the_coder_with_the_reason():
    m = Scripted(code=[ZERO, RIGHT], orch=[{"action": "coder"}, {"action": "coder", "message": "lọc đúng tên trong ĐẦU VÀO"}])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    second = m.prompts("code")[1]
    assert "the result is 0" in second and "lọc đúng tên trong ĐẦU VÀO" in second
    assert [t["verdict"]["source"] for t in st.log["turns"]] == ["checks", "judge"]


def test_three_rejected_turns_abstain_and_respect_the_budgets():
    m = Scripted(code=[LOOK])                                                  # never sets `answer`
    st = _st()
    assert _orch(m, max_turns=3, coder_steps=2).run(st) == {"type": "abstain"}
    assert len(st.log["turns"]) == 3 and all(len(t["steps"]) == 2 for t in st.log["turns"])
    decides = [e for e in st.log["orchestrator"] if "decide" in e]
    assert len(decides) == 3                                                   # after the spec + after turns 1 and 2


def test_guardrail_replaces_an_action_that_is_not_allowed():
    m = Scripted(code=[RIGHT], orch=[{"action": "run_sql"}])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    d = next(e for e in st.log["orchestrator"] if "decide" in e)
    assert d["action"] == "coder" and d["by"] == "default"


def test_explainer_rerun_is_offered_once():
    judge = {"verdict": "revise", "target": "explainer", "evidence": "bn cai", "problem": "đọc sai"}
    m = Scripted(code=[RIGHT], judge=[judge], orch=[{"action": "coder"}, {"action": "explainer", "message": "đọc lại"},
                                                   {"action": "explainer"}])
    st = _st()
    assert _orch(m, max_turns=3).run(st) == {"type": "abstain"}
    decides = [e for e in st.log["orchestrator"] if "decide" in e]
    assert "explainer" in decides[1]["allowed"] and "explainer" not in decides[2]["allowed"]
    assert decides[2]["action"] == "explainer" or decides[2]["by"] == "default"
    reruns = [e for e in st.log["orchestrator"] if e.get("agent") == "explainer" and e.get("rerun")]
    assert len(reruns) == 1 and "đọc lại" in m.prompts("explain")[-1]


def test_orchestrator_may_abstain_after_a_rejection():
    m = Scripted(code=[ZERO], orch=[{"action": "coder"}, {"action": "abstain", "reason": "cùng một lỗi"}])
    st = _st()
    assert _orch(m).run(st) == {"type": "abstain"} and len(st.log["turns"]) == 1


def test_counted_names_become_the_count():
    names = "```python\nd = tables['don']\nanswer = d[d['Tên hàng'] == 'Bánh tiramisu']['Mã đơn'].tolist()\n```"
    m = Scripted(code=[names], explain=[{"spec": {**SPEC, "measure": "count"}}],
                 type=[{"answer_type": "number", "measure": "count", "quote": "bn cai"}])
    assert _orch(m).run(_st()) == {"type": "number", "answer": 2}


def test_a_table_is_judged_by_its_structure_not_refused_outright():
    from datamate.answer.interpreter import interpret
    ok = {"columns": ["Khách", "Nợ"], "rows": [["A", 10], ["B", 5]]}
    assert interpret(ok, "table", SimpleNamespace(k=2, periods=[], measure="money")).issues == []
    assert interpret(ok, "table", SimpleNamespace(k=3, periods=[], measure="money")).issues == ["table_rows"]
    ragged = {"columns": ["Khách", "Nợ"], "rows": [["A"], ["B", 5]]}
    assert interpret(ragged, "table", None).issues == ["risky_type"]
    assert interpret([["A", 1]], "table", None).issues == ["risky_type"]       # no column names to check


def test_a_repeated_block_is_not_rerun_and_the_next_call_is_sampled():
    m = Scripted(code=[LOOK, LOOK, RIGHT])
    st = _st()
    assert _orch(m, coder_steps=3).run(st) == {"type": "number", "answer": 2}
    steps = st.log["turns"][0]["steps"]
    assert len(steps) == 3 and "IDENTICAL to block 1" in steps[1]["error"] and steps[2]["answered"]
    assert any(k.get("temperature") for k in m.kwargs)            # the call after the repeat is sampled


# ------------------------------------------------------------------ engine reference, windows, tables alias, coverage guard
def test_adjacent_windows_get_the_same_length_but_month_comparisons_are_left_alone():
    from datamate.agents.explainer import _fix_adjacent_windows
    q = "Kể từ ngày 11/05/2026, 14 ngày đầu bán được nhiều hay ít hơn 14 ngày liền trước đó bao nhiêu phần trăm"
    ps = [{"role": "main", "start": "2026-05-11", "end": "2026-05-24"}, {"role": "compare", "start": "2026-05-04", "end": "2026-05-10"}]
    notes = []
    _fix_adjacent_windows(ps, q, notes)
    assert (ps[1]["start"], ps[1]["end"]) == ("2026-04-27", "2026-05-10") and notes
    ps = [{"role": "main", "start": "2025-09-01", "end": "2025-09-30"}, {"role": "compare", "start": "2025-08-01", "end": "2025-08-31"}]
    _fix_adjacent_windows(ps, "thang nay so voi thang truoc", [])
    assert ps[1]["start"] == "2025-08-01"


def test_engine_reference_validation_and_comparison():
    from datamate.engines.reference import agrees, build_reference, clean_call
    assert clean_call({"function": "os.system", "args": {}}) is None
    assert clean_call({"function": "debt_repaid", "args": {"start": "yesterday", "end": "2026-07-28"}}) is None
    tools = {"debt_repaid": lambda start, end, customer=None: 3118333,
             "top_items": lambda k, start, end, by="qty": [("A", 9.0), ("B", 7.0), ("C", 5.0)]}
    r = build_reference({"function": "debt_repaid", "args": {"start": "2026-07-01", "end": "2026-07-28"}}, tools)
    assert r is not None and agrees(r, 3118333) and agrees(r, 3118333.4) and not agrees(r, 4143333)
    assert build_reference({"function": "debt_repaid", "args": {"begin": "2026-07-01"}}, tools) is None    # wrong argument names
    assert build_reference({"function": "item_qty", "args": {"item": "x", "start": "2026-07-01", "end": "2026-07-02"}}, tools) is None
    top = build_reference({"function": "top_items", "args": {"k": 3, "start": "2026-03-01", "end": "2026-03-31"}}, tools)
    assert agrees(top, ["A", "B", "C"]) and not agrees(top, ["B", "A", "C"]) and agrees(top, 12)      # a shape it cannot compare


class _Tools:
    """Inputs stub: the real sandbox variables plus one engine function."""
    def variables(self, st):
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "debt_repaid": lambda start, end, customer=None: 7,
                "period_start": pd.Timestamp("2026-05-01"), "period_end": pd.Timestamp("2026-05-31")}


def test_a_coder_that_disagrees_with_the_engine_reference_is_sent_back_once():
    call = {"function": "debt_repaid", "args": {"start": "2026-05-01", "end": "2026-05-31"}}
    wrong = "```python\nanswer = len(tables['don']) + 2\n```"
    right = "```python\nanswer = debt_repaid(period_start, period_end)\n```"
    m = Scripted(code=[wrong, right], explain=[{"spec": {**SPEC, "engine_call": call}}], orch=[{"action": "coder"}])
    st = _st()
    o = _orch(m)
    o.inputs = _Tools()
    assert o.run(st) == {"type": "number", "answer": 7}
    assert "ENGINE REFERENCE" in m.prompts("code")[0] and "debt_repaid(" in m.prompts("code")[0]
    assert "0. FIRST call `debt_repaid(start='2026-05-01', end='2026-05-31')`" in m.prompts("code")[0]   # the plan did not name it
    assert any(e.get("plan_engine_step") for e in st.log["orchestrator"])
    assert st.log["turns"][0]["issues"] == ["engine_mismatch"] and st.log["turns"][0]["verdict"]["source"] == "checks"
    # a Coder that keeps its own value on the second turn is trusted
    m2 = Scripted(code=[wrong], explain=[{"spec": {**SPEC, "engine_call": call}}], orch=[{"action": "coder"}])
    o2 = _orch(m2)
    o2.inputs = _Tools()
    assert o2.run(_st()) == {"type": "number", "answer": 5}


def test_sales_orders_and_item_lines_are_also_keys_of_tables():
    orders = pd.DataFrame({"order": ["O1"], "date": pd.to_datetime(["2026-05-01"]), "valid": [True]})
    ctx = SimpleNamespace(orders=orders, items=None, sales=None, debt=None)
    st = SimpleNamespace(period=None, ctx=ctx, tables=TABLES, as_of="2026-05-31")
    v = Inputs(pd.DataFrame()).variables(st)
    assert v["tables"]["sales_orders"] is v["sales_orders"] and "don" in v["tables"] and "item_lines" not in v["tables"]


def test_unanswerable_is_bounced_once_more_when_the_data_covers_the_period():
    orders = pd.DataFrame({"order": ["O1", "O2"], "date": pd.to_datetime(["2026-05-02", "2026-05-03"]), "valid": [True, False]})
    base = {"tables": {"don": TABLES["don"].copy()}, "sales_orders": orders, "period_start": pd.Timestamp("2026-05-01"),
            "period_end": pd.Timestamp("2026-05-31")}
    r = Runner(base, answerable=True)
    unans = "answer = 'UNANSWERABLE: no table'"
    assert "Before concluding" in r.run(unans).error                       # evidence first
    cov = r.run(unans)
    assert not cov.answered and "1 valid orders" in cov.error             # then the facts: 1 valid order in the period
    assert r.run(unans).answered                                          # a Coder that insists is accepted
    r2 = Runner(base, answerable=False)
    r2.run(unans)
    assert r2.run(unans).answered                                         # the Explainer already doubted: no second bounce
    empty = Runner({**base, "period_start": pd.Timestamp("2027-01-01"), "period_end": pd.Timestamp("2027-01-31")}, answerable=True)
    empty.run(unans)
    assert empty.run(unans).answered                                      # the period holds no orders: genuinely unanswerable


def test_a_top_k_spec_typed_string_becomes_a_list():
    bad = {**SPEC, "answer_type": "string", "measure": "name", "k": 3}
    m = Scripted(explain=[{"spec": bad}], type=[{"answer_type": "string", "measure": "name", "quote": "bn cai"}])
    st = _st()
    ExplainerAgent(m, m, m, "RULEBOOK", pd.DataFrame()).run(st)
    assert st.brief.answer_type == "list" and any("string -> list" in a for a in st.brief.assumptions)
    one = {**SPEC, "answer_type": "string", "measure": "name", "k": 1}
    m1 = Scripted(explain=[{"spec": one}], type=[{"answer_type": "string", "measure": "name", "quote": "bn cai"}])
    st1 = _st()
    ExplainerAgent(m1, m1, m1, "RULEBOOK", pd.DataFrame()).run(st1)
    assert st1.brief.answer_type == "string"


def test_calendar_phrases_are_computed_in_code():
    from datamate.agents.periods import calendar_period, fix_calendar_period
    d = lambda q, a: (lambda r: r and (str(r[0]), str(r[1])))(calendar_period(q, a))
    # 2026-09-15 is a Tuesday: last week is Mon 09-07 .. Sun 09-13, this week Mon 09-14 .. today
    assert d("dthu trung bình trên một đơn tuần rồi là bao nhiu", "2026-09-15") == ("2026-09-07", "2026-09-13")
    assert d("doanh thu tuan nay", "2026-09-15") == ("2026-09-14", "2026-09-15")
    assert d("thang 6/2025 top 5 hang ban chay", "2026-07-01") == ("2025-06-01", "2025-06-30")
    assert d("Tháng 2 năm 2024 bán được bao nhiêu", "2026-07-01") == ("2024-02-01", "2024-02-29")
    assert d("quý 4/2025 có bao nhiêu đơn", "2026-07-03") == ("2025-10-01", "2025-12-31")
    assert d("doanh thu tháng 3", "2026-07-03") == ("2026-03-01", "2026-03-31")          # no year: this year, it is not after today
    assert d("doanh thu tháng 9", "2026-07-03") == ("2025-09-01", "2025-09-30")          # no year, after today: last year (§2)
    assert d("doanh thu tháng này", "2026-07-03") == ("2026-07-01", "2026-07-03")
    # not sure -> None: comparisons, balances, windows, Tết, two phrases
    for q in ("doanh thu tháng này so với tháng trước", "nợ đến hết ngày 04/12/2025 là bao nhiêu", "14 ngày đầu so với 14 ngày liền trước",
              "tuần này và tuần trước bán bao nhiêu", "doanh thu tết năm nay", "từ tháng 3 đến tháng 5/2025 bán bao nhiêu"):
        assert calendar_period(q, "2026-07-03") is None, q
    ps = [{"label": "x", "start": "2025-08-25", "end": "2025-09-01", "role": "main"}]
    notes = []
    fix_calendar_period(ps, "tuan truoc, dthu ca chieu tang hay giam bao nhiu % so vs ca sang", "2025-09-06", notes)
    assert (ps[0]["start"], ps[0]["end"]) == ("2025-08-25", "2025-08-31") and notes        # shift comparison, one calendar phrase
    two = [{"label": "a", "start": "2025-08-25", "end": "2025-09-01", "role": "main"}, {"label": "b", "start": "2025-08-18", "end": "2025-08-24", "role": "compare"}]
    fix_calendar_period(two, "tuan truoc", "2025-09-06", [])
    assert two[0]["end"] == "2025-09-01"                                                    # two periods: left to the model


class _Ledger:
    """Inputs stub with a sales_orders frame so the coverage checks can run."""
    def __init__(self, orders, ps="2026-05-01", pe="2026-05-31"):
        self.orders, self.ps, self.pe = orders, pd.Timestamp(ps), pd.Timestamp(pe) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    def variables(self, st):
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "sales_orders": self.orders, "period_start": self.ps, "period_end": self.pe}


def _orders(days, walk_in):
    return pd.DataFrame({"order": [f"O{i}" for i in range(len(days))], "date": pd.to_datetime(days), "valid": True, "walk_in": walk_in})


def test_a_period_outside_the_data_is_unanswerable_without_calling_the_coder():
    m = Scripted(code=[RIGHT], explain=[{"spec": {**SPEC, "tables": {}}}])             # no other table with rows in May
    o = _orch(m)
    o.inputs = _Ledger(_orders(["2026-08-01", "2026-08-02"], False))                  # data starts after May 2026
    st = _st()
    got = o.run(st)
    assert got["type"] == "unanswerable" and "outside the dates" in got["reason"] and not m.prompts("code")
    o2 = _orch(Scripted(code=[RIGHT]))
    o2.inputs = _Ledger(_orders(["2026-05-02", "2026-08-02"], False))                 # the period has data: the Coder runs
    assert o2.run(_st()) == {"type": "number", "answer": 2}
    o3 = _orch(Scripted(code=[RIGHT]))                                                  # the spec's own table has May rows: the Coder runs
    o3.inputs = _Ledger(_orders(["2026-08-01", "2026-08-02"], False))
    assert o3.run(_st()) == {"type": "number", "answer": 2}


def test_a_customer_ranking_over_walk_ins_only_is_unanswerable():
    walk = "```python\nanswer = ' '.join(['khách', 'lẻ'])\nprint(tables['don'].shape)\n```"
    spec = {**SPEC, "answer_type": "string", "measure": "name", "k": 1, "entities": []}
    m = Scripted(code=[walk], explain=[{"spec": spec}], type=[{"answer_type": "string", "measure": "name", "quote": "bn cai"}])
    o = _orch(m)
    o.inputs = _Ledger(_orders(["2026-05-02", "2026-05-03"], True))
    got = o.run(_st())
    assert got["type"] == "unanswerable" and "walk-in" in got["reason"]
    m2 = Scripted(code=[walk], explain=[{"spec": spec}], type=[{"answer_type": "string", "measure": "name", "quote": "bn cai"}])
    o2 = _orch(m2, max_turns=1)
    o2.inputs = _Ledger(_orders(["2026-05-02", "2026-05-03"], [True, False]))        # a real customer exists: no shortcut
    assert o2.run(_st()) == {"type": "abstain"}


def test_missing_fields_mark_the_spec_possibly_unanswerable():
    from datamate.agents.explainer import repair_spec
    s, errs, warns, notes = repair_spec({**SPEC, "missing_fields": ["giới tính khách"]}, TABLES, "2026-05-31")
    assert s.answerable is False and "giới tính khách" in s.unanswerable_reason and notes
    s2, *_ = repair_spec({**SPEC, "missing_fields": []}, TABLES, "2026-05-31")
    assert s2.answerable is True


def test_order_column_names_and_negative_voucher_lines():
    from datamate.data.status import find_order_col
    from datamate.engines.sales import build_sales
    df = pd.DataFrame({"Đơn số": ["A", "A", "B"], "Đơn giá": [1.0, 2.0, 3.0], "Mã KH": ["x", "y", "z"], "Tên hàng": ["a", "b", "c"]})
    assert find_order_col(df) == "Đơn số"
    for name in ("HĐ số", "Bill no", "Số đơn", "Mã đơn hàng", "Invoice No"):
        assert find_order_col(pd.DataFrame({name: ["A", "A", "B"]})) == name, name
    assert find_order_col(pd.DataFrame({"Đơn giá": ["1", "2"]})) is None


def test_a_payments_table_is_not_taken_as_the_order_header():
    from datamate.engines.orders import build_orders
    from datamate.engines.sales import SalesView
    ids = [f"O{i}" for i in range(6) for _ in range(2)]
    lines = pd.DataFrame({"order": ids, "date": pd.to_datetime(["2026-05-01"] * 12), "amount": 1.0, "valid": True, "order_ok": True, "item": True})
    sales_tbl = pd.DataFrame({"Đơn số": ids, "Khách": ["A", "A", "B", "B", "C", "C", "", "", "E", "E", "F", "F"], "Kênh": "zalo"})
    pay = pd.DataFrame({"Ma don": [f"O{i}" for i in range(4)], "Tiền thu": 5.0})           # one row per PAID order: 4 of 6 orders
    biz = SimpleNamespace(tables={"BanHang": sales_tbl, "PhieuThu": pay}, readme="")
    out = build_orders(biz, SalesView(lines, "BanHang", "amount", "Đơn số", "date", "heuristic"))
    assert out is not None and len(out) == 6 and "Kênh" in out.columns and "Tiền thu" not in out.columns       # header from the lines table


def test_the_item_name_column_is_not_taken_for_customer_names():
    from datamate.engines.debt import _name_col
    n = 60
    items = [f"Bia Sài Gòn {i} lon" for i in range(n)]
    people = [f"Nguyễn Văn {chr(65 + i % 26)}{i}" for i in range(n)]
    df = pd.DataFrame({"Mã đơn hàng": [f"O{i}" for i in range(n)], "Tên hàng hóa": items})
    assert _name_col(df, "") is None                                             # no people in this table
    df["Khách hàng (tên)"] = people
    assert _name_col(df, "") == "Khách hàng (tên)"


def test_engine_hint_is_derived_from_the_spec_and_shown_to_the_coder():
    from datamate.engines.routing import hint_note, route
    tools = {"item_qty": lambda item, start, end: 31.0, "item_base_unit": lambda item: "kg",
             "top_items": lambda k, start, end, by="qty": [("A", 9.0), ("B", 7.0)],
             "top_customers": lambda k=None, start=None, end=None: [("An", 100)],
             "sales_metrics": lambda start, end: {"revenue": 1000, "orders": 4, "avg_order": 250}}
    def S(**kw):
        base = dict(answer_type="number", measure="quantity", k=None, entities=[], rewritten_question="x",
                    periods=[{"role": "main", "start": "2026-05-01", "end": "2026-05-31"}])
        return SimpleNamespace(**{**base, **kw})
    item = [{"value": "Nếp cái hoa vàng", "role": "item"}]
    h = route(S(entities=item), tools, "2026-05-01", "2026-05-31")
    assert h.call["function"] == "item_qty" and h.value == 31.0 and "kg" in h.note and "ENGINE HINT" in hint_note(h)
    assert h.agrees_with(31) and not h.agrees_with(186)
    assert route(S(answer_type="list", k=2), tools, "2026-05-01", "2026-05-31").call["function"] == "top_items"
    assert route(S(answer_type="string", measure="money", k=1, rewritten_question="Which customer spent the most"), tools,
                 "2026-05-01", "2026-05-31").call["function"] == "top_customers"
    assert route(S(measure="orders"), tools, "2026-05-01", "2026-05-31") is None            # whole-shop shapes were worse than the Coder
    assert route(S(entities=item, measure="money"), tools, "2026-05-01", "2026-05-31") is None
    assert route(S(entities=item, rewritten_question="What is the ending inventory quantity of X?"), tools, "2026-05-01", "2026-05-31") is None
    assert route(S(entities=item, rewritten_question="Tháng 5 bán bao nhiêu"), tools, "2026-05-01", "2026-05-31", "tháng 5 bán bao nhiêu") is None
    assert route(S(entities=item), tools, None, None) is None                                     # no period
    assert route(S(entities=item, periods=[{"role": "main"}, {"role": "compare"}]), tools, "2026-05-01", "2026-05-31") is None
    assert route(S(entities=item + item), tools, "2026-05-01", "2026-05-31") is None              # two entities: not a plain item question


class _Hinted:
    def __init__(self):
        self.calls = []

    def variables(self, st):
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "item_qty": lambda item, start, end: 2.0,
                "period_start": pd.Timestamp("2026-05-01"), "period_end": pd.Timestamp("2026-05-31")}


def test_the_hint_reaches_the_task_and_is_logged_without_deciding_the_verdict():
    m = Scripted(code=[RIGHT])
    o = _orch(m)
    o.inputs = _Hinted()
    st = _st()
    assert o.run(st) == {"type": "number", "answer": 2}
    assert "ENGINE HINT" in m.prompts("code")[0] and "item_qty" in m.prompts("code")[0]
    assert st.log["routing"][0]["agrees"] is True and len(st.log["turns"]) == 1


# ------------------------------------------------------------------ stuck Coder: guard resubmission, same error twice
class _OrdersInputs:
    """Inputs stub: the real sandbox variables plus `sales_orders` (so the `~valid` guard can fire)."""
    def variables(self, st):
        so = pd.DataFrame({"order": ["O1", "O2", "O3"], "date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]),
                           "valid": [True, False, True], "status_std": ["completed", "cancelled", "completed"]})
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "sales_orders": so,
                "period_start": pd.Timestamp("2026-05-01"), "period_end": pd.Timestamp("2026-05-31")}


def test_a_block_the_guard_refused_once_runs_when_sent_again():
    # the guard says "send the same block again"; before, the repeat check refused the resubmission as IDENTICAL
    block = "```python\nso = sales_orders\nanswer = int((~so['valid']).sum())\n```"
    m = Scripted(code=[block])
    st = _st()
    o = _orch(m)
    o.inputs = _OrdersInputs()
    assert o.run(st) == {"type": "number", "answer": 1}
    steps = st.log["turns"][0]["steps"]
    assert len(steps) == 2 and steps[0].get("refused") and "send the same block again" in steps[0]["error"]
    assert steps[1]["answered"] and "IDENTICAL" not in (steps[1]["error"] or "")


def test_the_same_error_twice_ends_the_turn_early_and_says_why():
    bad = ["```python\nprint(tables['nope'])\n```", "```python\nx = 1\nprint(tables['nope'])\n```",
           "```python\ny = 2\nprint(tables['nope'])\n```", LOOK]
    m = Scripted(code=bad)
    st = _st()
    assert _orch(m, max_turns=1, coder_steps=6).run(st) == {"type": "abstain"}
    turn = st.log["turns"][0]
    assert len(turn["steps"]) == 3                                      # not 6: two repeats of the same error end the turn
    assert any(e.get("stuck") == 2 for e in st.log["orchestrator"])
    assert turn["verdict"]["source"] == "no_answer" and "ended after 3 blocks" in turn["verdict"]["feedback"]
    assert sum(1 for k in m.kwargs if k.get("temperature")) >= 1       # the call after a repeated error is sampled


def test_the_plan_starts_with_the_engine_call_only_when_it_does_not_name_it():
    from datamate.agents.coder import engine_first
    from datamate.engines.reference import Reference
    ref = Reference({"function": "top_items", "args": {"k": 5, "start": "2026-06-01", "end": "2026-06-30"}}, [("A", 9)])
    plan, added = engine_first("1. group item_lines by item\n2. sum qty_base", ref)
    assert added and plan.startswith("0. FIRST call `top_items(k=5,") and plan.endswith("2. sum qty_base")
    assert engine_first("1. call top_items(5, ...) and keep the names", ref) == ("1. call top_items(5, ...) and keep the names", False)
    assert engine_first("", ref)[0].startswith("0. FIRST call") and engine_first("1. x", None) == ("1. x", False)


def test_unambiguous_shapes_are_fixed_and_the_rest_get_a_concrete_reshape_hint():
    from datamate.answer.evaluator import reshape_hint
    from datamate.answer.interpreter import interpret
    top = SimpleNamespace(k=2, periods=[], measure="quantity")
    plain = SimpleNamespace(k=None, periods=[], measure="money")
    # one (name, number) pair for a one-value question: the half it asks for
    assert interpret([("Chị Lan", 3200000)], "string", plain).raw == "Chị Lan"
    assert interpret(("Chị Lan", 3200000), "number", plain).raw == 3200000
    # a ranking kept as a Series formatted as its amounts before; names for a top-k, amounts for period labels
    it = interpret(pd.Series([9, 7], index=["Bánh A", "Bánh B"]), "list", top)
    assert (it.kind, it.raw, it.issues) == ("list", ["Bánh A", "Bánh B"], [])
    assert interpret(pd.Series([5, 6], index=["2025-04", "2025-05"]), "list", plain).raw == [5, 6]
    assert list(interpret(pd.Series([5, 6], index=["Tiền mặt", "Chuyển khoản"]), "list", plain).raw) == [5, 6]  # unchanged
    # a DataFrame for a list question is judged as its rows
    df = pd.DataFrame({"item": ["Bánh A", "Bánh B"], "qty": [9, 7]})
    assert interpret(df, "list", top).raw == ["Bánh A", "Bánh B"]
    assert interpret(df, "list", plain).issues == ["type"]                 # names or amounts? not clear: still refused
    # the refusal says how to reshape instead of only "re-read the question"
    st, steps = _st(), [RunResult(code="answer = df", answered=True, value=None)]
    st.answer_type = "list"
    st.brief = SimpleNamespace(k=None)
    v = _ev({"verdict": "accept"}).evaluate(st, "C", steps, True, None, Interpretation("table", None, ["type"]), None)
    assert "only change the form" in v.feedback and "[r[0] for r in rows]" in v.feedback
    assert "ONE name" in reshape_hint("string", "list") and "len(...)" in reshape_hint("number", "list")



def test_records_and_rows_of_dicts_do_not_crash_the_interpreter():
    # dev5-coderphase q_33e8f7506387: a KeyError: 0 here abstained the whole question (pipeline catch-all), not the turn
    from datamate.answer.interpreter import interpret
    top = SimpleNamespace(k=2, periods=[], measure="quantity")
    recs = [{"item": "Bánh A", "q": 9}, {"item": "Bánh B", "q": 7}]
    assert interpret(recs, "list", top).raw == ["Bánh A", "Bánh B"]
    assert interpret([{"item": "Bánh A"}, {"item": "Bánh B"}], "list", top).raw == ["Bánh A", "Bánh B"]
    it = interpret({"columns": ["item", "q"], "rows": recs}, "table", top)
    assert it.kind == "table" and "risky_type" in it.issues                  # judged, not crashed
    assert interpret([{"item": "khách lẻ", "q": 9}], "list", SimpleNamespace(k=None, periods=[], measure="money")).issues


def test_an_interpreter_error_rejects_the_turn_not_the_question(monkeypatch):
    import datamate.agents.orchestrator as orch
    calls = {"n": 0}

    def flaky(raw, expected, spec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError(0)
        from datamate.answer.interpreter import interpret
        return interpret(raw, expected, spec)
    monkeypatch.setattr(orch, "interpret", flaky)
    m = Scripted(code=[RIGHT])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    assert any("interpreter_error" in e for e in st.log["orchestrator"]) and len(st.log["turns"]) == 2
