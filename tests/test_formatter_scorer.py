import pandas as pd

from datamate.answer.formatter import format_answer
from datamate.eval.scorer import is_correct, norm_str


def test_number_rounding_and_numpy():
    import numpy as np
    assert format_answer("number", np.float64(48250000.0)) == {"type": "number", "answer": 48250000}
    assert format_answer("number", 18.456) == {"type": "number", "answer": 18.46}
    assert format_answer("number", 368727.27) == {"type": "number", "answer": 368727}  # average of money -> whole đồng
    assert format_answer("percent", -12.4549) == {"type": "percent", "answer": -12.45}


def test_date_table_unanswerable():
    assert format_answer("date", pd.Timestamp("2026-08-15"))["answer"] == "2026-08-15"
    assert format_answer("date", "2025-09")["answer"] == "2025-09"
    df = pd.DataFrame({"kh": ["A"], "no": [3200000.0]})
    assert format_answer("table", df)["answer"] == {"columns": ["kh", "no"], "rows": [["A", 3200000]]}
    assert format_answer("string", "UNANSWERABLE: thiếu giá vốn")["type"] == "unanswerable"


def test_scorer_string_alias_and_type():
    gold = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0042"]}
    assert is_correct({"type": "string", "answer": "sp0042"}, gold)
    assert not is_correct({"type": "string", "answer": "Sting dau 330ml"}, gold)
    assert not is_correct({"type": "number", "answer": 1}, gold)
    assert norm_str("  Cà-phê  SỮA!") == "cà phê sữa"


def test_plumbing_fixes():
    from datamate.answer.interpreter import _heuristic
    # dict-form unanswerable
    out = format_answer("string", {"type": "unanswerable", "reason": "thiếu cột"})
    assert out == {"type": "unanswerable", "reason": "thiếu cột"} and _heuristic({"type": "unanswerable"}) == "string"
    # rows -> table, not list
    rows = [["A", 1.0], ["B", 2.0]]
    assert _heuristic(rows) == "table"
    assert format_answer("table", rows)["answer"]["rows"] == [["A", 1], ["B", 2]]
    assert format_answer("table", [{"kh": "A", "no": 3}])["answer"]["columns"] == ["kh", "no"]
    # float-ified month strings
    assert _heuristic("2025.0-10.0") == "date"
    assert format_answer("date", "2025.0-10.0")["answer"] == "2025-10"


def test_a_list_returned_as_name_amount_pairs_keeps_the_half_asked_for():
    # dev300 q_33e8f7506387: the right top 5 with their quantities was refused, the hand-rebuilt ranking was wrong (abstain)
    from types import SimpleNamespace
    from datamate.answer.interpreter import interpret
    pairs = [["Cà phê sữa (M)", 308], ["Trà đào cam sả (M)", 244]]
    r = interpret(pairs, "list", SimpleNamespace(k=2, measure="quantity", periods=[]))
    assert (r.kind, r.raw, r.issues) == ("list", ["Cà phê sữa (M)", "Trà đào cam sả (M)"], [])
    months = [["2025-04", 120], ["2025-05", 150]]                         # one number per month: the numbers
    assert interpret(months, "list", SimpleNamespace(k=None, measure="money", periods=[])).raw == [120, 150]
    unclear = interpret(pairs, "list", SimpleNamespace(k=None, measure="money", periods=[]))
    assert "type" in unclear.issues                                      # names or amounts? the check stays
