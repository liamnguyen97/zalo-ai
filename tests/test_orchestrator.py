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
    # DATAMATE_DECIDE=model: the model decides and its message reaches the Coder; the rule's choice is logged next to it
    m = Scripted(code=[ZERO, RIGHT], orch=[{"action": "coder"}, {"action": "coder", "message": "lọc đúng tên trong ĐẦU VÀO"}])
    st = _st()
    assert _orch(m, decide_mode="model").run(st) == {"type": "number", "answer": 2}
    second = m.prompts("code")[1]
    assert "the result is 0" in second and "lọc đúng tên trong ĐẦU VÀO" in second
    assert [t["verdict"]["source"] for t in st.log["turns"]] == ["checks", "judge"]
    after = [e for e in st.log["orchestrator"] if str(e.get("decide", "")).startswith("after_evaluator")]
    assert after[0]["by"] == "model" and after[0]["rule_would"] == "coder"


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
    assert _orch(m, max_turns=3, decide_mode="model", trust_repeat=False).run(st) == {"type": "abstain"}
    decides = [e for e in st.log["orchestrator"] if "decide" in e]
    assert "explainer" in decides[1]["allowed"] and "explainer" not in decides[2]["allowed"]
    assert decides[2]["action"] == "explainer" or decides[2]["by"] == "default"
    reruns = [e for e in st.log["orchestrator"] if e.get("agent") == "explainer" and e.get("rerun")]
    assert len(reruns) == 1 and "đọc lại" in m.prompts("explain")[-1]


def test_orchestrator_may_abstain_after_a_rejection():
    m = Scripted(code=[ZERO], orch=[{"action": "coder"}, {"action": "abstain", "reason": "cùng một lỗi"}])
    st = _st()
    assert _orch(m, decide_mode="model").run(st) == {"type": "abstain"} and len(st.log["turns"]) == 1


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


def test_a_call_with_a_formal_argument_is_repaired_and_a_condition_drops_it():
    """dev300 2b4ca48e19: 120 of 242 engine calls were dropped whole for one argument the function does not take."""
    from datamate.agents.coder import engine_first
    from datamate.engines.reference import reference_note, resolve_reference
    tools = {"sales_metrics": lambda start, end: {"revenue": 7_100_000, "orders": 66, "avg_order": 107_748},
             "item_qty": lambda item, start, end: 31, "item_base_unit": lambda item: "kg",
             "purchase_avg_price": lambda item, start, end, unit=None: 200_556}
    day = {"start": "2026-05-11", "end": "2026-05-11"}
    ref, why = resolve_reference({"function": "sales_metrics", "args": {**day, "metric": "order_count"}}, tools, "number")
    assert why == "" and ref.value == 66 and ref.key == "orders"                      # q_4362: the engine says 66, the Coder said 0
    assert ref.expr == "sales_metrics(start='2026-05-11', end='2026-05-11')['orders']" and ref.text.endswith("-> 66")
    assert "['orders']" in reference_note(ref) and "FIRST call `sales_metrics(start='2026-05-11', end='2026-05-11')['orders']`" in engine_first("", ref)[0]
    avg, _ = resolve_reference({"function": "sales_metrics", "args": {**day, "metric": "avg_revenue_per_order", "valid_only": True}}, tools, "number")
    assert avg.value == 107_748
    nep = {"function": "item_qty", "args": {"item": "Nếp cái hoa vàng", **day, "unit": "KG"}}
    assert resolve_reference(nep, tools, "number")[0].value == 31                     # the base unit, compared folded
    # conditions the function cannot express: the whole shop's figure would contradict the right answer
    for bad in ({"metric": "cancelled_orders"}, {"shift": "afternoon"}, {"channel": "zalo_oa"}, {"valid_only": False}):
        r, why = resolve_reference({"function": "sales_metrics", "args": {**day, **bad}}, tools, "number")
        assert r is None and "does not take it" in why, bad
    assert resolve_reference({"function": "item_qty", "args": {"item": "x", **day, "unit": "bao"}}, tools, "number")[0] is None
    # a repaired call is a cross-check of a plain number only: a revenue is not the answer of a percent change
    r, why = resolve_reference({"function": "sales_metrics", "args": {**day, "metric": "revenue"}}, tools, "percent")
    assert r is None and "number answer" in why
    assert resolve_reference({"function": "sales_metrics", "args": {**day, "metric": "revenue"}}, tools, None)[0] is None
    # a call the function takes as it is: unchanged, for any answer type (as before); `unit` is a real argument of purchase_avg_price
    assert resolve_reference({"function": "sales_metrics", "args": day}, tools, "percent")[0].value["orders"] == 66
    pap = resolve_reference({"function": "purchase_avg_price", "args": {"item": "x", **day, "unit": "thùng"}}, tools, "number")[0]
    assert pap.call["args"]["unit"] == "thùng" and pap.key is None


def test_a_dropped_engine_call_is_logged_once_with_its_reason():
    call = {"function": "debt_repaid", "args": {"start": "2026-05-01", "end": "2026-05-31", "shift": "sáng"}}
    m = Scripted(code=[RIGHT], explain=[{"spec": {**SPEC, "engine_call": call}}])
    st = _st()
    o = _orch(m)
    o.inputs = _Tools()
    o.run(st)
    dropped = [e for e in st.log["orchestrator"] if "engine_call_dropped" in e]
    assert len(dropped) == 1 and "shift='sáng'" in dropped[0]["engine_call_dropped"] and dropped[0]["call"] == call
    assert st.engine_ref is None and "ENGINE REFERENCE" not in m.prompts("code")[0]


class _Tools:
    """Inputs stub: the real sandbox variables plus one engine function."""
    def variables(self, st):
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "debt_repaid": lambda start, end, customer=None: 7,
                "item_qty": lambda item, start, end: 7,
                "period_start": pd.Timestamp("2026-05-01"), "period_end": pd.Timestamp("2026-05-31")}


def test_a_coder_that_disagrees_with_the_engine_reference_is_sent_back_once():
    # a quantity function: the question asks "bn cái" (a reference of another kind is dropped, engines/reference.kind_conflict)
    call = {"function": "item_qty", "args": {"item": "Bánh tiramisu", "start": "2026-05-01", "end": "2026-05-31"}}
    wrong = "```python\nanswer = len(tables['don']) + 2\n```"
    right = "```python\nanswer = item_qty('Bánh tiramisu', period_start, period_end)\n```"
    m = Scripted(code=[wrong, right], explain=[{"spec": {**SPEC, "engine_call": call}}], orch=[{"action": "coder"}])
    st = _st()
    o = _orch(m)
    o.inputs = _Tools()
    assert o.run(st) == {"type": "number", "answer": 7}
    assert "ENGINE REFERENCE" in m.prompts("code")[0] and "item_qty(" in m.prompts("code")[0]
    assert "0. FIRST call `item_qty(item='Bánh tiramisu', start='2026-05-01', end='2026-05-31')`" in m.prompts("code")[0]   # the plan did not name it
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
                 "2026-05-01", "2026-05-31", "Tháng 5 khách nào mua nhiều tiền nhất?").call["function"] == "top_customers"
    assert route(S(measure="orders"), tools, "2026-05-01", "2026-05-31") is None            # whole-shop shapes were worse than the Coder
    assert route(S(entities=item, measure="money"), tools, "2026-05-01", "2026-05-31") is None
    assert route(S(entities=item, rewritten_question="What is the ending inventory quantity of X?"), tools, "2026-05-01", "2026-05-31",
                 "Tồn kho Nếp cái hoa vàng cuối ngày hôm nay là bao nhiêu kg?") is None
    assert route(S(entities=item, rewritten_question="Tháng 5 bán bao nhiêu"), tools, "2026-05-01", "2026-05-31", "tháng 5 bán bao nhiêu") is None
    assert route(S(entities=item), tools, None, None) is None                                     # no period
    assert route(S(entities=item, periods=[{"role": "main"}, {"role": "compare"}]), tools, "2026-05-01", "2026-05-31") is None
    assert route(S(entities=item + item), tools, "2026-05-01", "2026-05-31") is None              # two entities: not a plain item question
    # the item's BASE unit named as a second entity is no condition (dev300 q_1885: `cái` in `Đơn vị tính`, role item)
    kg = {"value": "Kg", "column": "Đơn vị tính", "role": "item"}
    assert route(S(entities=item + [kg]), tools, "2026-05-01", "2026-05-31").call["function"] == "item_qty"
    assert route(S(entities=[kg] + item), tools, "2026-05-01", "2026-05-31").call["function"] == "item_qty"   # by value, not position
    bao = {"value": "bao", "column": "Đơn vị tính", "role": "item"}                              # another unit: a real condition
    assert route(S(entities=item + [bao]), tools, "2026-05-01", "2026-05-31") is None
    no_units = {k: v for k, v in tools.items() if k != "item_base_unit"}
    assert route(S(entities=item + [kg]), no_units, "2026-05-01", "2026-05-31") is None           # no base unit known: as before


def test_engine_hints_read_the_structured_spec_and_the_original_question_not_the_rewrite():
    """dev300 q_4333ad0589a1: the English rewrite said "customers", so a debt question got the REVENUE ranking `top_customers`."""
    from datamate.engines.routing import route
    tools = {"item_qty": lambda item, start, end: 31.0, "top_customers": lambda k=None, start=None, end=None: [("An", 100)]}
    def S(**kw):
        base = dict(answer_type="table", measure="money", k=3, entities=[], rewritten_question="x", engine_call=None, rule_sections=[],
                    periods=[{"role": "main", "start": "2025-01-01", "end": "2026-05-20"}])
        return SimpleNamespace(**{**base, **kw})
    debt_q = "cho bảng 3 khách còn nợ nhìu nhất tính đến hết ngày 20/05/2026, kèm số tiền còn nợ"
    rewrite = "List the top 3 customers with the highest outstanding debt as of 2026-05-20"
    a, b = "2025-01-01", "2026-05-20"
    # each of the three gates is enough: the question's words, the spec's engine_call, the rulebook sections it cites
    assert route(S(rewritten_question=rewrite), tools, a, b, debt_q) is None
    debt_call = {"function": "debt_by_customer", "args": {"day": b, "k": 3}}
    assert route(S(rewritten_question=rewrite, engine_call=debt_call), tools, a, b, "3 khách mua nhiều tiền nhất") is None
    assert route(S(rule_sections=["4. Công nợ"]), tools, a, b, "3 khách mua nhiều tiền nhất") is None
    # a sales section cited too keeps the hint: the model cites §5 "Nhập trừ bán (quy về cùng đơn vị)" for a unit conversion
    sales = route(S(rule_sections=["1. Đơn hàng và doanh thu", "5. Kho, nhập hàng, giá vốn"]), tools, a, b, "3 khách mua nhiều tiền nhất")
    assert sales.call["function"] == "top_customers"
    # the rewrite's language no longer matters: a Vietnamese rewrite gets the same hint
    assert route(S(rewritten_question="3 khách hàng có doanh thu cao nhất"), tools, a, b, "3 khách hàng có tổng tiền mua cao nhất là ai?").call
    assert route(S(), tools, a, b, "3 nhóm khách nào mua nhiều tiền nhất") is None                 # a customer group is not a customer
    assert route(S(), tools, a, b, "3 nhân viên nào bán nhiều tiền nhất") is None
    assert route(S(), tools, a, b, None) is None                                                     # no question: no customer reading
    item = [{"value": "Nếp cái hoa vàng", "role": "item"}]
    qty = dict(answer_type="number", measure="quantity", k=None, entities=item)
    assert route(S(**qty), tools, a, b, "Tháng 5 bán được bao nhiêu kg Nếp cái hoa vàng?").call["function"] == "item_qty"
    assert route(S(**qty), tools, a, b, "Số nhập trừ số bán của Nếp cái hoa vàng tháng 5 là bao nhiêu kg?") is None
    assert route(S(**qty), tools, a, b, "Trung bình mỗi ngày bán bao nhiêu kg Nếp cái hoa vàng?") is None
    assert route(S(**qty), tools, a, b, "Tháng 5 bán Nếp cái hoa vàng được bao nhiêu tiền?") is None   # money, though the spec says quantity
    tole = [{"value": "Tôn lạnh màu xanh", "role": "item"}]                                         # "tôn" in a name is not "tồn"
    assert route(S(**{**qty, "entities": tole}), tools, a, b, "Tháng 5 bán được bao nhiêu tấm Tôn lạnh màu xanh?").call
    same = {"function": "item_qty", "args": {"item": "Nếp cái hoa vàng", "start": a, "end": b, "unit": "bao"}}   # dropped, same function
    assert route(S(**qty, engine_call=same), tools, a, b, "Tháng 5 bán được bao nhiêu kg Nếp cái hoa vàng?").call["function"] == "item_qty"


class _Hinted:
    def __init__(self):
        self.calls = []

    def variables(self, st):
        return {"tables": {k: v.copy() for k, v in TABLES.items()}, "item_qty": lambda item, start, end: 2.0,
                "period_start": pd.Timestamp("2026-05-01"), "period_end": pd.Timestamp("2026-05-31")}


class _HintedKg(_Hinted):
    def variables(self, st):
        return {**super().variables(st), "item_base_unit": lambda item: "cái"}


def test_an_enforced_hint_sends_a_differing_coder_back_once():
    """dev300: the item_qty hint equalled the gold on 16 of 16 questions, the Coder overrode it on 4 (Nếp cái: 62 for 31)."""
    five = "```python\nanswer = 5\n```"
    m = Scripted(code=[five, RIGHT])
    o = _orch(m)
    o.inputs = _HintedKg()
    st = _st()
    assert o.run(st) == {"type": "number", "answer": 2}
    t0 = st.log["turns"][0]
    assert t0["issues"] == ["hint_mismatch"] and t0["verdict"]["source"] == "checks" and t0["verdict"]["target"] == "coder"
    assert "item_qty(item='Bánh tiramisu', start='2026-05-01', end='2026-05-31') -> 2.0" in t0["verdict"]["feedback"]
    assert st.log["routing"][0]["enforced"] is True
    # a Coder that keeps its own value is trusted (said once)
    m2 = Scripted(code=[five])
    o2 = _orch(m2)
    o2.inputs = _HintedKg()
    assert o2.run(_st()) == {"type": "number", "answer": 5}
    # not a listed function: logged only, as before
    m3 = Scripted(code=[five])
    o3 = _orch(m3, enforce_hints=())
    o3.inputs = _HintedKg()
    st3 = _st()
    assert o3.run(st3) == {"type": "number", "answer": 5} and st3.log["turns"][0]["issues"] == []
    assert "enforced" not in st3.log["routing"][0]


def test_an_item_qty_hint_is_not_enforced_when_the_question_asks_in_another_unit():
    five = "```python\nanswer = 5\n```"
    for unit, said in (("thùng", False), ("Cái", True)):        # the engine counts in the base unit ("cái")
        m = Scripted(code=[five, RIGHT], explain=[{"spec": {**SPEC, "unit": unit}}])
        o = _orch(m)
        o.inputs = _HintedKg()
        st = _st()
        o.run(st)
        assert (st.log["turns"][0]["issues"] == ["hint_mismatch"]) is said, unit


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


# ------------------------------------------------------------------ P4 / P5 (ledger #12): rule table, trusted repeat, no judge on engine
def _t(source, target="coder", issues=()):
    return {"verdict": {"source": source, "decision": "revise", "target": target}, "issues": list(issues)}


def test_the_rule_table_after_a_rejected_turn():
    from datamate.agents.orchestrator import NO_ANSWER_TWICE, decide_by_rule
    both, no_rerun = ["coder", "abstain", "explainer"], ["coder", "abstain"]
    first = decide_by_rule([_t("checks", issues=["zero"])], both)
    assert (first.action, first.by, first.message) == ("coder", "rule", "")                  # no free text from a model
    assert decide_by_rule([_t("checks", issues=["zero"]), _t("checks", issues=["type"])], both).action == "coder"
    twice = [_t("checks", issues=["walk_in"]), _t("checks", issues=["walk_in"])]
    d = decide_by_rule(twice, both)
    assert d.action == "explainer" and "walk_in" in d.message                              # the Coder cannot fix it: the spec
    assert decide_by_rule(twice, no_rerun).action == "abstain"
    assert decide_by_rule([_t("no_answer")], both).action == "coder"
    assert decide_by_rule([_t("no_answer"), _t("no_answer")], both).message == NO_ANSWER_TWICE
    assert decide_by_rule([_t("no_answer"), _t("no_answer")], no_rerun).action == "coder"
    assert decide_by_rule([_t("judge", "explainer")], both).action == "explainer"
    assert decide_by_rule([_t("judge", "explainer")], no_rerun).action == "coder"
    assert decide_by_rule([_t("judge", "coder"), _t("judge", "coder")], both).action == "coder"


def test_rule_mode_asks_no_model_after_a_rejected_turn():
    # the first reply is the legacy decision after the spec; the second would be the model's after the rejected turn
    m = Scripted(code=[ZERO, RIGHT], orch=[{"action": "coder"}, {"action": "abstain", "message": "lời nhắn của model"}])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    assert len(m.prompts("orch")) == 1                                   # only the legacy decision after the spec
    after = [e for e in st.log["orchestrator"] if str(e.get("decide", "")).startswith("after_evaluator")]
    assert len(after) == 1 and after[0]["by"] == "rule" and after[0]["action"] == "coder" and "rule_would" not in after[0]
    second = m.prompts("code")[1]
    assert "the result is 0" in second and "lời nhắn của model" not in second


def test_a_value_the_judge_rejected_and_the_coder_returned_unchanged_is_accepted():
    # dev5-coderphase2 q_33e8f7506387: the right top 5 rejected three times by the judge, the question abstained
    judge = {"verdict": "revise", "target": "coder", "evidence": "bn cai", "problem": "đếm sai"}
    m = Scripted(code=[RIGHT], judge=[judge])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}
    assert [t["verdict"]["source"] for t in st.log["turns"]] == ["judge", "repeat_after_veto"]
    assert len(m.prompts("judge")) == 1                                  # the judge is heard once
    m2 = Scripted(code=[RIGHT], judge=[judge])
    st2 = _st()
    assert _orch(m2, trust_repeat=False).run(st2) == {"type": "abstain"}  # switch off: the old behaviour
    assert len(m2.prompts("judge")) == 3 and any(e.get("repeat_after_veto") for e in st2.log["orchestrator"])   # still logged


def test_no_judge_when_the_value_equals_an_exact_engine_function():
    from datamate.answer.evaluator import JUDGE_SYSTEM
    call = {"function": "item_qty", "args": {"item": "Bánh tiramisu", "start": "2026-05-01", "end": "2026-05-31"}}
    right = "```python\nanswer = item_qty('Bánh tiramisu', period_start, period_end)\n```"
    m = Scripted(code=[right], explain=[{"spec": {**SPEC, "engine_call": call}}])
    o = _orch(m, skip_judge_on_engine=True)
    o.inputs = _Tools()
    st = _st()
    assert o.run(st) == {"type": "number", "answer": 7}
    assert st.log["turns"][0]["verdict"]["source"] == "engine_agrees" and not m.prompts("judge")
    m2 = Scripted(code=[right], explain=[{"spec": {**SPEC, "engine_call": call}}])
    o2 = _orch(m2)                                                        # off by default since the dev300 measurement
    o2.inputs = _Tools()
    st2 = _st()
    assert o2.run(st2) == {"type": "number", "answer": 7} and len(m2.prompts("judge")) == 1
    assert any(e.get("engine_confirms") for e in st2.log["orchestrator"])   # logged with the switch off too
    assert "ENGINE VALUE" in m2.prompts("judge")[0] and "item_qty(" in m2.prompts("judge")[0]
    # the routing hint counts as well; a value the engine does not confirm still goes to the judge
    m3 = Scripted(code=[RIGHT])
    o3 = _orch(m3, skip_judge_on_engine=True)
    o3.inputs = _Hinted()                                                 # item_qty -> 2.0, the Coder counts 2
    st3 = _st()
    assert o3.run(st3) == {"type": "number", "answer": 2} and st3.log["turns"][0]["verdict"]["source"] == "engine_agrees"
    assert "Check only" in JUDGE_SYSTEM and "never ask to add amounts" in JUDGE_SYSTEM and "(5)" not in JUDGE_SYSTEM


def test_confirms_is_strict_where_agrees_is_lenient():
    import numpy as np
    from datamate.engines.reference import Reference, agrees, confirms
    num = Reference({"function": "debt_repaid", "args": {}}, 3118333)
    assert confirms(num, 3118333) and confirms(num, 3118333.4) and confirms(num, np.int64(3118333))
    assert not confirms(num, 4143333) and agrees(num, "abc") and not confirms(num, "abc")
    top = Reference({"function": "top_items", "args": {"k": 3}}, [("A", 9.0), ("B", 7.0), ("C", 5.0)])
    assert confirms(top, ["A", "B", "C"]) and not confirms(top, ["B", "A", "C"])
    assert agrees(top, ["A", "B"]) and not confirms(top, ["A", "B"])          # a prefix is not the same top 3
    assert agrees(top, 12) and not confirms(top, 12)
    assert confirms(Reference({"function": "top_customers", "args": {}}, [("An", 100)]), "an")
    metrics = Reference({"function": "sales_metrics", "args": {}}, {"revenue": 1000, "orders": 4, "avg_order": 250})
    assert agrees(metrics, 250) and not confirms(metrics, 250)               # whole-shop figures ignore the question's conditions
    assert confirms(Reference({"function": "item_qty_change", "args": {}}, {"before": 10, "after": 12, "pct": 20.0}), 20.0)


def test_an_unanswerable_the_code_concluded_is_submitted_without_the_judge():
    # dev300 v3.5: 2 of the 3 questions the review damaged were a right UNANSWERABLE it sent back (ledger #12)
    st, steps = _st(), [RunResult(code="answer = 'UNANSWERABLE: x'", answered=True, value="UNANSWERABLE: x")]
    it, line = Interpretation("unanswerable", "UNANSWERABLE: x"), {"type": "unanswerable", "reason": "x"}
    m = Scripted(judge=[{"verdict": "revise", "evidence": "bn cai", "problem": "có dữ liệu"}])
    v = Evaluator(m).evaluate(st, "C", steps, True, "UNANSWERABLE: x", it, line)
    assert (v.decision, v.source) == ("unanswerable", "unanswerable_direct") and not m.prompts("judge")
    m2 = Scripted(judge=[{"verdict": "revise", "evidence": "bn cai", "problem": "có dữ liệu"}])
    v2 = Evaluator(m2, judge_unanswerable=True).evaluate(st, "C", steps, True, "UNANSWERABLE: x", it, line)
    assert v2.decision == "revise" and len(m2.prompts("judge")) == 1           # DATAMATE_JUDGE_UNANSWERABLE=1: the old way
    # end to end: the Runner asks for evidence once, then the conclusion is submitted
    first = "```python\nprint(list(tables))\nanswer = 'UNANSWERABLE: no such column'\n```"
    again = "```python\nprint(tables['don'].columns.tolist())\nanswer = 'UNANSWERABLE: no such column'\n```"
    m3 = Scripted(code=[first, again])
    st3 = _st()
    got = _orch(m3).run(st3)
    assert got["type"] == "unanswerable" and st3.log["turns"][0]["verdict"]["source"] == "unanswerable_direct"
    assert not m3.prompts("judge")


def test_the_judge_reads_the_verified_facts_and_may_quote_them():
    st, steps = _st(), [RunResult(code="answer = n", output="2", answered=True, value=2)]
    st.facts = "# VERIFIED FACTS (counted by the system from the data; counts, not the answer)\n- tháng 5: 1 valid orders.\n\n"
    it, line = Interpretation("number", 2), {"type": "number", "answer": 2}
    judge = {"verdict": "revise", "evidence": "1 valid orders", "problem": "2 lớn hơn số đơn hợp lệ"}
    m = Scripted(judge=[judge])
    v = Evaluator(m).evaluate(st, "C", steps, True, 2, it, line)
    assert "VERIFIED FACTS" in m.prompts("judge")[0] and v.decision == "revise" and v.source == "judge"
    assert v.found_in == "facts"                                              # logged: quotes from FACTS widen the filter
    m2 = Scripted(judge=[judge])
    v2 = Evaluator(m2, judge_facts=False).evaluate(st, "C", steps, True, 2, it, line)
    assert "VERIFIED FACTS" not in m2.prompts("judge")[0] and v2.source == "judge_ignored"   # evidence nowhere to be found


def test_the_judge_reads_the_periods_code_computed_only_when_the_flag_is_on():
    from datetime import date

    from datamate.agents.explainer import Spec
    from datamate.agents.question import PeriodMention
    from datamate.context import Period, Resolved
    st, steps = _st(), [RunResult(code="answer = n", output="2", answered=True, value=2)]
    st.named_periods = [PeriodMention("month 3/2025", date(2025, 3, 1), date(2025, 3, 31))]
    st.period = Resolved(date(2026, 8, 21), [Period("March 2025", date(2025, 3, 1), date(2025, 3, 31), "main")])
    st.brief = Spec(rewritten_question="q", open_issues=["periods: the question names month 4/2025 (2025-04-01..2025-04-30) and no spec period equals it", "entities: x"])
    it, line = Interpretation("number", 2), {"type": "number", "answer": 2}
    judge = {"verdict": "revise", "evidence": "2025-03-01 .. 2025-03-31", "problem": "the Coder used another month"}
    m = Scripted(judge=[judge])
    v = Evaluator(m, judge_periods=True).evaluate(st, "C", steps, True, 2, it, line)
    p = m.prompts("judge")[0]
    assert "# PERIODS (computed by code" in p and "month 3/2025: 2025-03-01 .. 2025-03-31" in p and "main 'March 2025'" in p
    assert "no spec period equals it" in p and "entities: x" not in p              # only the period issues
    assert v.decision == "revise" and v.found_in == "facts"                         # the dates are evidence the judge may quote
    m2 = Scripted(judge=[judge])
    Evaluator(m2).evaluate(st, "C", steps, True, 2, it, line)
    assert "PERIODS (computed" not in m2.prompts("judge")[0]                        # off by default


def test_the_code_audits():
    from datamate.answer.audit import entity_absent, profit_bound, string_literals
    natri = ("mask = joined['Hoạt chất'].str.lower().str.contains('natri|clor', na=False)\n"
             "print(f'Total revenue for Natri Clorid: {answer}')\nprint('Natri clorid')")
    assert "natri clorid" not in string_literals(natri)                       # f-strings and print() only echo the name
    spec = SimpleNamespace(entities=[{"value": "Natri clorid", "role": "item"}])
    assert entity_absent(spec, [natri]) == ["Natri clorid"]                  # dev300 q_0f6233b739e9
    assert entity_absent(spec, ["d = df[df['Hoạt chất'] == 'Natri clorid']"]) == []
    assert entity_absent(spec, ["x = (", "d = df[df['Hoạt chất'] == 'natri clorid']"]) == []   # a broken block hides nothing
    assert entity_absent(SimpleNamespace(entities=[]), [natri]) == [] and entity_absent(spec, []) == []
    v = {"sales_metrics": lambda s, e: {"revenue": 314535600, "orders": 2795, "avg_order": 112535},
         "period_start": pd.Timestamp("2025-01-01"), "period_end": pd.Timestamp("2025-01-31 23:59:59")}
    q = "Xem giúp trừ hết chi phí rồi thì tháng 1/2025 còn lãi ròng bao nhiêu?"   # dev300 q_40de33fe6cdc, gold -62,275,893
    assert "larger in size" in profit_bound(q, "number", -2292011721, v)
    assert profit_bound(q, "number", -62275893, v) == "" and profit_bound(q, "percent", -2292011721, v) == ""
    assert profit_bound("thang 1 lai rong bao nhieu", "number", 5e9, v)        # no diacritics
    assert profit_bound("tháng 1 còn lại bao nhiêu tiền hàng", "number", 5e9, v) == ""   # "còn lại" is not "còn lãi"


def test_audits_are_logged_in_shadow_and_said_once_when_on():
    substr = "```python\nd = tables['don']\nn = int(d['Tên hàng'].str.contains('tiramisu').sum())\nanswer = n\n```"
    m = Scripted(code=[substr])
    st = _st()
    assert _orch(m).run(st) == {"type": "number", "answer": 2}                # shadow (default): decides nothing
    t = st.log["turns"][0]
    assert t["audit"] == {"entity_absent": ["Bánh tiramisu"]} and t["verdict"]["source"] == "judge"
    m2 = Scripted(code=[substr])
    st2 = _st()
    assert _orch(m2, audit_mode="on").run(st2) == {"type": "number", "answer": 2}
    assert [t["verdict"]["source"] for t in st2.log["turns"]] == ["checks", "judge"]   # sent back once, then trusted
    assert "Bánh tiramisu" in st2.log["turns"][0]["verdict"]["feedback"]
    m3 = Scripted(code=[substr])
    st3 = _st()
    _orch(m3, audit_mode="off").run(st3)
    assert "audit" not in st3.log["turns"][0]


# ------------------------------------------------------------------ audits of the periods (error analysis, "Review")
def _mention(label, a, b):
    from datetime import date

    from datamate.agents.question import PeriodMention
    return PeriodMention(label, date.fromisoformat(a), date.fromisoformat(b))


def _periods(*rows):
    from datetime import date

    from datamate.context import Period
    return [Period(lbl, date.fromisoformat(a), date.fromisoformat(b), role) for lbl, a, b, role in rows]


def test_period_mismatch_wants_the_periods_handed_to_the_coder_to_equal_the_ones_the_question_names():
    from datamate.answer.audit import period_mismatch
    jan = [_mention("month 1/2026", "2026-01-01", "2026-01-31"), _mention("month 12/2025", "2025-12-01", "2025-12-31")]
    q = "T1/2026 so với T12/2025 doanh thu tăng bao nhiêu %"
    quarter = _periods(("q1", "2026-01-01", "2026-03-31", "main"), ("dec", "2025-12-01", "2025-12-31", "compare"))
    why = period_mismatch(q, jan, quarter)
    assert "month 1/2026 2026-01-01..2026-01-31" in why and "main 2026-01-01..2026-03-31" in why and "month 12/2025" not in why
    exact = _periods(("jan", "2026-01-01", "2026-01-31", "main"), ("dec", "2025-12-01", "2025-12-31", "compare"))
    assert period_mismatch(q, jan, exact) == ""
    assert "none" in period_mismatch(q, jan, [])                                # the spec has no period at all
    # a balance: the period handed to the Coder starts at the first day of the data and may be wider than the phrase
    bal = [_mention("month 5/2026", "2026-05-01", "2026-05-31")]
    assert period_mismatch("tồn kho đến hết tháng 5/2026", bal, _periods(("all", "2025-01-01", "2026-05-31", "main"))) == ""
    # to-date against the same days of the previous month: the comparison period is cut short on purpose
    q2 = "So với cùng kỳ tháng trước, doanh thu từ đầu tháng này đến hết hôm nay tăng hay giảm bao nhiêu %"
    named = [_mention("last month", "2026-04-01", "2026-04-30"), _mention("month to date", "2026-05-01", "2026-05-24")]
    got = _periods(("mtd", "2026-05-01", "2026-05-24", "main"), ("same days", "2026-04-01", "2026-04-24", "compare"))
    assert period_mismatch(q2, named, got) == ""


def test_date_literal_finds_dates_the_coder_typed_outside_every_period():
    from datamate.answer.audit import code_dates, date_literal
    ps = _periods(("may", "2026-05-01", "2026-05-31", "main"))
    named = [_mention("month 5/2026", "2026-05-01", "2026-05-31")]
    q, as_of = "tháng 5 bán được bao nhiêu cái", "2026-07-22"
    typed = "d = tables['don']\nm = d[(d['Ngày'] >= pd.Timestamp('2025-05-01')) & (d['Ngày'] <= pd.Timestamp('2025-05-31 23:59:59'))]\nanswer = len(m)"
    assert "2025-05-01" in date_literal(q, named, ps, typed, as_of)
    assert "2025-03-01" in date_literal(q, named, ps, "m = d[d['Ngày'] >= pd.Timestamp(2025, 3, 1)]", as_of)       # a call with ints
    assert date_literal(q, named, ps, "m = d[(d['Ngày'] >= period_start) & (d['Ngày'] <= period_end)]\nanswer = len(m)", as_of) == ""
    assert date_literal(q, named, ps, "m = d[d['Ngày'] < pd.Timestamp('2026-06-01')]", as_of) == ""                   # end + 1 day
    assert date_literal(q, named, ps, "print(f'dates 2025-05-01'); print('2025-05-01')", as_of) == ""                 # only echoes
    assert date_literal(q, named, ps, "d = d[d['Ngày'] >= '1900-01-01']", as_of) == ""                                # a sentinel
    assert date_literal("tồn kho đến hết tháng 5", named, ps, typed, as_of) == ""                                    # a balance: wide bounds
    assert date_literal("tháng 5 sau ngày 01/05/2025 bán bao nhiêu", named, ps, typed, as_of) != ""                  # one date of the question, not both
    assert date_literal("so với ngày 31/05/2025 và 01/05/2025 tháng 5 bán bao nhiêu", named, ps, typed, as_of) == ""  # the question writes them
    assert date_literal(q, [], [], typed, as_of) == ""                                                               # nothing to compare with
    assert code_dates("x = (") == set() and len(code_dates("a = '2025-05-01'; b = '01/06/2025'; c = '2025-05-01 08:00'")) == 2


def test_a_period_mismatch_goes_back_to_the_explainer_once_when_the_audit_is_on():
    def run(**kw):
        m = Scripted(code=[RIGHT])
        st = _st()
        st.named_periods = [_mention("month 4/2026", "2026-04-01", "2026-04-30")]       # the spec says May: nobody covers April
        return m, st, _orch(m, **kw).run(st)
    m0, st0, got0 = run()                                                                # shadow: logged, decides nothing
    assert got0 == {"type": "number", "answer": 2} and "month 4/2026" in st0.log["turns"][0]["audit"]["period_mismatch"]
    assert st0.log["turns"][0]["verdict"]["source"] == "judge"
    m1, st1, got1 = run(audit_mode="on")
    assert got1 == {"type": "number", "answer": 2}
    assert [t["verdict"]["source"] for t in st1.log["turns"]] == ["checks", "judge"]     # sent back once, then trusted
    first = st1.log["turns"][0]["verdict"]
    assert first["target"] == "explainer" and "month 4/2026" in first["feedback"] and "2026-05-01..2026-05-31" in first["feedback"]
    assert len(m1.prompts("explain")) > len(m0.prompts("explain"))                       # the Explainer ran again
    m2, st2, _ = run(audit_mode="on", explainer_reruns=0)                                # no re-run left: the Coder cannot fix it, so not said
    assert [t["verdict"]["source"] for t in st2.log["turns"]] == ["judge"]
    m3, st3, _ = run(audit_mode="on", audit_skip=("period_mismatch",))
    assert [t["verdict"]["source"] for t in st3.log["turns"]] == ["judge"] and "period_mismatch" in st3.log["turns"][0]["audit"]


def test_date_literal_is_logged_but_enforced_only_when_it_is_not_skipped():
    typed = ("```python\nd = tables['don']\nn = int(((d['Tên hàng'] == 'Bánh tiramisu') & (d['Ngày'] >= '2025-05-01')).sum())\n"
             "answer = n\n```")
    m = Scripted(code=[typed])
    st = _st()
    _orch(m, audit_mode="on").run(st)                                                    # default skip: date_literal is log-only
    assert "2025-05-01" in st.log["turns"][0]["audit"]["date_literal"] and st.log["turns"][0]["verdict"]["source"] == "judge"
    m2 = Scripted(code=[typed, RIGHT])
    st2 = _st()
    assert _orch(m2, audit_mode="on", audit_skip=()).run(st2) == {"type": "number", "answer": 2}
    assert [t["verdict"]["source"] for t in st2.log["turns"]] == ["checks", "judge"]
    assert st2.log["turns"][0]["verdict"]["target"] == "coder" and "period_start" in st2.log["turns"][0]["verdict"]["feedback"]


def test_the_judge_gets_room_for_its_whole_reply(monkeypatch):
    """dev300: max_tokens=800 cut 8 of 287 judge replies before the closing brace, read as unreadable (= accept)."""
    from datamate.config import Settings
    seen = {}

    def fake_chat(model, system, user, stage="", max_tokens=None, context=None, **kw):
        seen[stage] = max_tokens
        return '{"verdict": "accept"}'
    monkeypatch.setattr("datamate.answer.evaluator.chat", fake_chat)
    st, steps = _st(), [RunResult(code="answer = n", output="2", answered=True, value=2)]
    it, line = Interpretation("number", 2), {"type": "number", "answer": 2}
    assert Evaluator(Scripted()).evaluate(st, "C", steps, True, 2, it, line).decision == "accept" and seen["judge"] == 1500
    Evaluator(Scripted(), max_tokens=900).evaluate(st, "C", steps, True, 2, it, line)
    assert seen["judge"] == 900 and Settings().judge_max_tokens == 1500


def test_a_judge_reply_cut_by_the_cap_is_read_up_to_the_cut(monkeypatch):
    """B4: v2 writes evidence before problem / fix and reads the complete fields of a cut reply; the verbatim filter still decides."""
    from datamate.answer.evaluator import JUDGE_SYSTEM, JUDGE_SYSTEM_V1
    sent = {}

    def judge_says(text):
        def fake_chat(model, system, user, stage="", max_tokens=None, context=None, **kw):
            sent["system"] = system
            return text
        monkeypatch.setattr("datamate.answer.evaluator.chat", fake_chat)
    st, steps = _st(), [RunResult(code="answer = n", output="2", answered=True, value=2)]
    st.facts = "# VERIFIED FACTS\n- tháng 5: 1 valid orders.\n"
    it, line = Interpretation("number", 2), {"type": "number", "answer": 2}
    judge_says('{"verdict": "revise", "target": "coder", "evidence": "1 valid orders", "problem": "2 is more than the valid ord')
    v = Evaluator(Scripted()).evaluate(st, "C", steps, True, 2, it, line)
    assert sent["system"] == JUDGE_SYSTEM and v.decision == "revise" and v.found_in == "facts" and "reply cut" in v.feedback
    judge_says('{"verdict": "revise", "target": "coder", "evidence": "1 valid ord')               # the quote itself was cut
    assert Evaluator(Scripted()).evaluate(st, "C", steps, True, 2, it, line).source == "judge_ignored"
    judge_says('{"verdict": "revise", "target": "coder", "evidence": "1 valid orders"} {"verdict": "accept"}')    # broken, not cut
    assert Evaluator(Scripted()).evaluate(st, "C", steps, True, 2, it, line).source == "judge_ignored"
    judge_says('{"verdict": "revise", "target": "coder", "evidence": "1 valid orders", "problem": "2 is more')
    v1 = Evaluator(Scripted(), judge_format="v1").evaluate(st, "C", steps, True, 2, it, line)      # as before: unreadable = accept
    assert sent["system"] == JUDGE_SYSTEM_V1 and v1.source == "judge_ignored" and "unreadable" in v1.feedback
    fmt = lambda sysp: sysp[sysp.index("Return only JSON"):]                                        # noqa: E731
    assert fmt(JUDGE_SYSTEM).index('"evidence"') < fmt(JUDGE_SYSTEM).index('"problem"')
    assert fmt(JUDGE_SYSTEM_V1).index('"problem"') < fmt(JUDGE_SYSTEM_V1).index('"evidence"')


def test_a_revise_whose_evidence_pieces_are_in_the_facts_stands():
    """B1: "VERIFIED FACTS: ... 19... ENGINE VALUE: ... -> 32" is in no text as a whole (dev300 q_c4eee7e4ca8d: a right diagnosis, dropped)."""
    st, steps = _st(), [RunResult(code="answer = n", output="140", answered=True, value=140)]
    st.facts = "# VERIFIED FACTS\n- valid sales lines of item 'Nếp cái hoa vàng' in the main period: 19\n"
    it, line = Interpretation("number", 140), {"type": "number", "answer": 140}
    ev = "VERIFIED FACTS: valid sales lines of item 'Nếp cái hoa vàng' in the main period: 19... the code returned 140"
    judge = {"verdict": "revise", "target": "coder", "evidence": ev, "problem": "140 kg from 19 lines"}
    v = Evaluator(Scripted(judge=[judge])).evaluate(st, "C", steps, True, 140, it, line)
    assert v.decision == "revise" and v.found_in == "facts_fragments"
    strict = Evaluator(Scripted(judge=[judge]), evidence="strict").evaluate(st, "C", steps, True, 140, it, line)
    assert strict.source == "judge_ignored"                                               # as before
    # pieces found only in the question or the code do not count (format complaints about right answers got through that way)
    q_only = {**judge, "evidence": "ORIGINAL QUESTION: tiramisu thang 5... bn cai"}
    assert Evaluator(Scripted(judge=[q_only])).evaluate(st, "C", steps, True, 140, it, line).source == "judge_ignored"
    # boilerplate of the facts is no evidence: the header, or a phrase every fact line has, without a number of the facts
    st.facts = "# VERIFIED FACTS (counted by the system from the data; counts, not the answer)\n" + st.facts.split("\n", 1)[1]
    for ev in ("VERIFIED FACTS (counted by the system from the data; counts, not the answer)... the answer is not a date",
               "the main period... the format is wrong"):
        v = Evaluator(Scripted(judge=[{**judge, "evidence": ev}])).evaluate(st, "C", steps, True, 140, it, line)
        assert v.source == "judge_ignored", ev
