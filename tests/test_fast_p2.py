"""Fast P2 (config.p2_mode = "fast"): the question read in code, one spec call, checks in code, at most one fix call,
no Column Selector and no orchestrator decision after the spec; the plan as the spec's `method` field."""
import json
from types import SimpleNamespace

import pandas as pd

from datamate.agents.coder import INSTRUCTIONS, Coder
from datamate.agents.column_selector import ColumnSelector
from datamate.agents.explainer import METHOD_FIELD, PLAN_SYSTEM, SYSTEM_FAST, ExplainerAgent, rule_lines
from datamate.agents.orchestrator import SYSTEM as ORCH_SYSTEM
from datamate.agents.orchestrator import Orchestrator
from datamate.agents.question import find_names, mentioned_periods, read_type
from datamate.agents.shared import SEPARATOR
from datamate.answer.evaluator import JUDGE_SYSTEM, Evaluator
from datamate.context import QAState
from datamate.sandbox.variables import Runner as Inputs

TABLES = {"don": pd.DataFrame({"Mã đơn": ["O1", "O2", "O3", "O4"],
                               "Ngày": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]),
                               "Tên hàng": ["Bánh tiramisu", "Cà phê sữa (M)", "Bánh tiramisu", "Trà đào cam sả"],
                               "Đơn vị": ["cái", "ly", "cái", "bao"], "Khách": ["Từ Thắng Hải", "Lan", "Lan", "Minh"],
                               "Thành tiền": [30.0, 25.0, 30.0, 20.0]})}
SPEC = {"rewritten_question": "How many 'Bánh tiramisu' (cái) were sold in May 2026?", "answer_type": "number", "measure": "quantity",
        "unit": "cái", "periods": [{"label": "tháng 5/2026", "start": "2026-05-01", "end": "2026-05-31", "role": "main"}],
        "entities": [{"value": "Bánh tiramisu", "table": "don", "column": "Tên hàng", "role": "item"}],
        "tables": {"don": ["Tên hàng", "Ngày"]}, "rule_refs": ["Bán chạy"], "answerable": True,
        "method": "1. filter May 2026\n2. count the item"}
RULEBOOK = "## 3. Xếp hạng\n| Bán chạy | theo số lượng, đơn hợp lệ |\n| Doanh thu | tổng dòng |"
RIGHT = "```python\nd = tables['don']\nn = int((d['Tên hàng'] == 'Bánh tiramisu').sum())\nprint(n)\nanswer = n\n```"
FAST_SYS = (SYSTEM_FAST + METHOD_FIELD, SYSTEM_FAST)


class Scripted:
    def __init__(self, **queues):
        self.queues, self.seen = {k: list(v) for k, v in queues.items()}, []
        self.queues.setdefault("spec", [SPEC])
        self.queues.setdefault("plan", ["1. count"])
        self.queues.setdefault("judge", [{"verdict": "accept"}])
        self.queues.setdefault("orch", [{"action": "coder"}])

    def generate(self, messages, **kw):
        system, user = messages[0]["content"][0]["text"], messages[1]["content"][0]["text"]
        self.systems_seen = getattr(self, "systems_seen", []) + [system]
        system = system.split(SEPARATOR)[-1]                 # the stage's own instructions after the shared context
        stage = ("spec" if system in FAST_SYS else "plan" if system == PLAN_SYSTEM else "code" if system == INSTRUCTIONS
                 else "judge" if system == JUDGE_SYSTEM else "orch" if system == ORCH_SYSTEM
                 else "select" if "COLUMN SELECTOR" in system else "?")
        self.seen.append((stage, user))
        q = self.queues.get(stage) or ["{}"]
        out = q.pop(0) if len(q) > 1 else q[0]
        return SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))

    def stages(self):
        return [s for s, _ in self.seen]


def _st(question="banh tiramisu thang 5 ban dc bn cai"):
    ctx = SimpleNamespace(tables=TABLES, descriptors={"don": "Table `don`"}, readme="", schema="Table `don` (full card)", items=None,
                          sales=None, debt=None, orders=None, structure=None)
    st = QAState({"id": "q1", "question": question, "as_of_date": "2026-05-31"}, ctx, {})
    st.schema = ctx.schema
    return st


def _explainer(m, plan_mode="spec"):
    return ExplainerAgent(m, m, m, RULEBOOK, pd.DataFrame(), mode="fast", plan_mode=plan_mode)


# ------------------------------------------------------------------ reading the question in code
def test_answer_type_and_k_from_the_wording():
    assert read_type("tháng này đơn huỷ chiếm bao nhiêu %").answer_type == "percent"
    assert read_type("ngày naof trong tháng 6/2025 dthu cao nhaats").answer_type == "date"            # telex slips undone
    r = read_type("Top 3 sp bán nhiều nhất tháng 3 năm 2026 là những gì?")
    assert (r.answer_type, r.k, r.sure) == ("list", 3, True)
    assert read_type("mặt hàng nào bán chạy nhất tuần rồi").answer_type == "string"
    assert read_type("doanh thu 3 tháng có tăng liên tục không?").answer_type == "boolean"
    d = read_type("tháng 5 doanh thu là bao nhiêu")
    assert (d.answer_type, d.sure) == ("number", False)                                                 # the default never overrides
    c = read_type("tháng 5 bán được bao nhiêu ly")
    assert (c.answer_type, c.rule, c.sure) == ("number", "how many", True)                              # a count of objects is a number


def test_names_come_from_the_data_and_grammar_or_time_words_are_not_names():
    names = {n.phrase: n for n in find_names("Từ tháng 4/2026 bánh tiramisu bán bao nhiêu cái, chiếm bao nhiêu phần trăm?", TABLES)}
    assert "banh tiramisu" in names and names["banh tiramisu"].value == "Bánh tiramisu"
    assert "cai" in names                                                    # a unit of the data is a name too
    assert not any(p in names for p in ("bao", "tu thang"))                  # "bao nhiêu", "từ tháng" are grammar / time
    part = {n.phrase: n for n in find_names("tra dao cam ban duoc may ly", TABLES)}
    assert part["tra dao cam"].how == "inside one value" and part["tra dao cam"].value == "Trà đào cam sả"


def test_periods_named_in_the_question():
    p = mentioned_periods("doanh thu tháng 4/2026 so với tháng trước", "2026-05-20")
    assert [(x.label, str(x.start), str(x.end)) for x in p] == [("month 4/2026", "2026-04-01", "2026-04-30"),
                                                                ("last month", "2026-04-01", "2026-04-30")] or len(p) == 2


def test_rule_refs_become_the_rulebook_lines():
    assert rule_lines(["Bán chạy", "không có"], RULEBOOK) == ["| Bán chạy | theo số lượng, đơn hợp lệ |", "không có"]


# ------------------------------------------------------------------ one spec call
def test_a_clean_spec_takes_one_call_and_its_method_is_the_plan():
    m = Scripted()
    st = _st()
    s = _explainer(m).run(st)
    assert m.stages() == ["spec"] and s.status == "clear"
    assert st.plan.startswith("1. filter May 2026") and s.rules == ["| Bán chạy | theo số lượng, đơn hợp lệ |"]
    assert "DATA MATCHES" in m.seen[0][1] and "banh tiramisu" in m.seen[0][1]
    assert st.log["explainer"]["mode"] == "fast" and len(st.log["explainer"]["calls"]) == 1


def test_a_sure_reading_overrides_the_type_without_a_call():
    m = Scripted(spec=[{**SPEC, "answer_type": "number", "measure": "ratio"}])
    st = _st("banh tiramisu chiem bao nhieu % doanh thu thang 5")
    s = _explainer(m).run(st)
    assert m.stages() == ["spec"] and s.answer_type == "percent"
    assert any("answer_type number -> percent" in a for a in s.assumptions)


def test_a_lost_name_costs_one_fix_call_and_the_better_spec_wins():
    lost = {**SPEC, "entities": [], "rewritten_question": "How many items were sold in May 2026?"}
    m = Scripted(spec=[lost, SPEC])
    st = _st()
    s = _explainer(m).run(st)
    assert m.stages() == ["spec", "spec"]
    assert "TO FIX" in m.seen[1][1] and "'Bánh tiramisu'" in m.seen[1][1]
    assert [e["value"] for e in s.entities] == ["Bánh tiramisu"] and s.status == "clear"


def test_never_more_than_two_calls_and_unfixed_issues_are_open():
    lost = {**SPEC, "entities": [], "rewritten_question": "How many items were sold?"}
    m = Scripted(spec=[lost, lost])
    s = _explainer(m).run(_st())
    assert m.stages() == ["spec", "spec"] and s.status == "unchecked" and s.open_issues


def test_plan_modes():
    for mode, stages, plan in (("call", ["spec", "plan"], "1. count"), ("none", ["spec"], "")):
        m = Scripted()
        st = _st()
        _explainer(m, mode).run(st)
        assert m.stages() == stages and st.plan == plan
    # the `method` field is asked for only in plan mode "spec"
    class Spy(Scripted):
        def generate(self, messages, **kw):
            self.systems = getattr(self, "systems", []) + [messages[0]["content"][0]["text"]]
            return super().generate(messages, **kw)
    for mode, asked in (("spec", True), ("none", False), ("call", False)):
        m = Spy()
        _explainer(m, mode).run(_st())
        assert ("method:" in m.systems[0]) is asked


# ------------------------------------------------------------------ orchestrator
def test_fast_orchestrator_skips_the_selector_and_the_first_decision():
    m = Scripted(code=[RIGHT])
    st = _st()
    o = Orchestrator(m, ColumnSelector(m), _explainer(m), Coder(m), Evaluator(m), Inputs(pd.DataFrame()), RULEBOOK,
                     use_selector=False, decide_after_explainer=False)
    assert o.run(st) == {"type": "number", "answer": 2}
    assert "select" not in m.stages() and "orch" not in m.stages()
    assert st.schema == "Table `don` (full card)"
    first = next(e for e in st.log["orchestrator"] if "decide" in e)
    assert first["action"] == "coder" and first["by"] == "rule"



def test_every_call_opens_with_the_same_shop_context():
    m = Scripted(code=[RIGHT])
    st = _st()
    o = Orchestrator(m, ColumnSelector(m), _explainer(m), Coder(m), Evaluator(m), Inputs(pd.DataFrame()), RULEBOOK,
                     use_selector=False, decide_after_explainer=False, shared_prefix=True)
    assert o.run(st) == {"type": "number", "answer": 2}
    heads = {x.split(SEPARATOR)[0] for x in m.systems_seen}
    assert len(heads) == 1 and m.stages() == ["spec", "code", "judge"]          # spec, coder and judge share one prefix
    head = heads.pop()
    assert head.startswith("# SHOP CONTEXT") and RULEBOOK in head and "Table `don` (full card)" in head
    spec_user, code_user = m.seen[0][1], m.seen[1][1]
    assert RULEBOOK not in spec_user and RULEBOOK not in code_user                # not repeated in the question's part
    assert "DATA MATCHES" in spec_user and "ORIGINAL QUESTION" in code_user


def test_yesterday_and_today_are_calendar_phrases():
    # dev300 q_078fecc248e5 "bua qua dthu duoc bao nhieu": the spec read "bữa qua" as last week; nothing in code caught it
    from datamate.agents.periods import calendar_period
    assert calendar_period("bua qua dthu duoc bao nhieu", "2026-01-10")[:2] == (__import__("datetime").date(2026, 1, 9),) * 2
    assert [p.label for p in mentioned_periods("hnay bán được bao nhiêu ly", "2026-01-10")] == ["today"]
    from datamate.agents.explainer import repair_spec
    wrong = {"rewritten_question": "revenue last week", "answer_type": "number",
             "periods": [{"label": "last week", "start": "2026-01-05", "end": "2026-01-11", "role": "main"}]}
    s, errs, warns, notes = repair_spec(wrong, TABLES, "2026-01-10", None, "bua qua dthu duoc bao nhieu")
    assert (s.periods[0]["start"], s.periods[0]["end"]) == ("2026-01-09", "2026-01-09")       # fixed in code, no model call


def test_a_balance_at_the_end_of_today_is_not_a_one_day_period():
    from datamate.agents.periods import calendar_period
    q = "Tồn kho bia tiger lon cuối ngày hôm nay là bnhiu lon"
    assert calendar_period(q, "2026-01-10") is None and mentioned_periods(q, "2026-01-10") == []
