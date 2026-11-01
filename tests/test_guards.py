import types

import pytest

from datamate.llm import _record, get_usage, reset_usage
from datamate.sandbox.guard import GuardedExecutor
from datamate.sandbox.variables import _end_of_day


class _Inner:
    def __init__(self):
        self.last = None

    def __call__(self, code):
        self.last = code
        return "ran"


def test_guard_blocks_overwriting_injected_data():
    g = GuardedExecutor(_Inner())
    assert g("df = tables['a']\nprint(len(df))") == "ran"
    for bad in ("tables = {'a': 1}", "calendar = None", "for tables in x: pass"):
        with pytest.raises(ValueError):
            g(bad)
    assert g("as_of_date = '2025-01-01'") == "ran"  # allowed
    assert g("not python (((") == "ran"  # syntax errors are left to the sandbox to report


def test_harness_appends_final_answer_for_answer_variable():
    inner = _Inner(); g = GuardedExecutor(inner)
    g("df = tables['a']\nanswer = int(df['x'].sum())")
    assert inner.last.rstrip().endswith("final_answer(answer)")
    g("answer = int(df['x'].sum())\nfinal_answer(answer)")  # model called it itself: not duplicated
    assert inner.last.count("final_answer(") == 1


def test_guard_rejects_made_up_constant_answers():
    g = GuardedExecutor(_Inner())
    # the exact biz_125 pattern: correct total printed, then a hard-coded number laundered through a variable
    with pytest.raises(ValueError):
        g("df = tables['a']\ntotal = df['x'].sum()\nprint(total)\ntotal_int = round(128470000.0)\nfinal_answer(total_int)")
    with pytest.raises(ValueError):
        g("df = tables['a']\nprint(df['x'].sum())\nanswer = 1284567890")
    with pytest.raises(ValueError):
        g("x = tables['a']\nfinal_answer([1, 2, 3])")


def test_guard_allows_computed_and_observed_answers():
    g = GuardedExecutor(_Inner())
    assert g("df = tables['a']\nanswer = round(df['x'].sum())") == "ran"             # computed
    assert g("answer = 629820000") == "ran"                                           # bare, value seen in an earlier step
    assert g("final_answer(17318236300)") == "ran"
    assert g("df = tables['a']\nanswer = 'UNANSWERABLE: thiếu cột'") == "ran"       # unanswerable is a literal by nature
    assert g("df = tables['a']\nanswer = {'type': 'unanswerable', 'reason': 'x'}") == "ran"


def test_literal_fallback_in_a_branch_is_allowed():
    g = GuardedExecutor(_Inner())
    assert g("df = tables['a']\nm = df['x'] > 1\nif not m.any():\n    answer = 0\nelse:\n    answer = int(df['x'].sum())") == "ran"
    with pytest.raises(ValueError):  # the unconditional guess is still refused
        g("df = tables['a']\nprint(df['x'].sum())\nif True:\n    pass\nanswer = 1284567890")


def test_constant_laundered_across_steps_is_rejected():
    g = GuardedExecutor(_Inner())
    g("guess = 123456")                                    # bare block: allowed, but `guess` is remembered as a literal
    with pytest.raises(ValueError):
        g("df = tables['a']\nprint(df['x'].sum())\nanswer = guess")


def test_usage_recording_with_and_without_details():
    reset_usage()
    u = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10, model_extra={"cost": 0.001},
                              prompt_tokens_details=types.SimpleNamespace(cached_tokens=64))
    _record(types.SimpleNamespace(usage=u))
    _record(types.SimpleNamespace(usage=None))  # e.g. vLLM mock without usage
    assert get_usage() == {"calls": 1, "input_tokens": 100, "output_tokens": 10, "cached_tokens": 64, "cost_usd": 0.001}


def test_period_end_includes_the_whole_last_day():
    import pandas as pd
    end = _end_of_day("2025-03-31")
    s = pd.Series([pd.Timestamp("2025-03-31"), pd.Timestamp("2025-03-31 23:59:59"), pd.Timestamp("2025-04-01")])
    assert ((s <= end).tolist(), (s < end).tolist()) == ([True, True, False], [True, True, False])


def test_branch_fallback_literal_does_not_make_answer_a_constant():
    """q_1885ab61433f / q_12bcf28ac9f0: `else: answer = "UNANSWERABLE..."` + closing `answer = answer` was refused 6 times."""
    inner = _Inner(); g = GuardedExecutor(inner)
    g("df = tables['a']\nif len(df):\n    answer = int(df['x'].sum())\nelse:\n    answer = 'UNANSWERABLE: thiếu'\n"
      "print(answer)\nanswer = answer")
    assert inner.last.rstrip().endswith("final_answer(answer)")
    g("df = tables['a']\nif len(df):\n    total = 0\nelse:\n    total = int(df['x'].sum())\nanswer = total")  # branch literal
    # an unconditional made-up number is still refused
    with pytest.raises(ValueError):
        g("df = tables['a']\nn = len(df)\nanswer = 128470000")


def _merge_guard(sales_table):
    import pandas as pd
    lines = pd.DataFrame({"order": ["O1"], "item": ["a"], "amount": [1.0], "Nhóm hàng": ["bia"]})
    g = GuardedExecutor(lambda code: "ran")
    g.tables, g.frames, g.sales_table = {"ban": lines, "nhap": lines, "phieu": lines, "sp": lines}, {"item_lines": lines}, sales_table
    return g


def test_merge_of_purchase_tables_is_not_refused_but_a_sales_merge_still_is():
    # biz_109 q_00a6e23cbacf: purchase lines + purchase slips (and 'Nhóm hàng' read from the product table) lost a block
    purchase = ("d = tables['nhap']\nd = d[d['x'] == 1]\nj = d.merge(tables['phieu'][['a', 'b']], on='a')\n"
                "g = tables['sp']['Nhóm hàng']")
    sales = "d = tables['ban']\nj = d.merge(tables['sp'][['a', 'Nhóm hàng']], on='a')"
    assert _merge_guard("ban")(purchase) == "ran"
    with pytest.raises(ValueError, match="item_lines"):
        _merge_guard("ban")(sales)
    pd_merge = "j = pd.merge(tables['ban'], tables['sp'][['a', 'Nhóm hàng']], on='a')"
    with pytest.raises(ValueError, match="item_lines"):
        _merge_guard("ban")(pd_merge)
    # sales table unknown: the old rule (any merge that reads a product attribute is refused once)
    with pytest.raises(ValueError, match="item_lines"):
        _merge_guard(None)(purchase)
