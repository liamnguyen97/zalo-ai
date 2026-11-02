"""Coder phase inputs computed in code: the rulebook rows of the question (GUIDELINE) and counts from the data (VERIFIED FACTS)."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from datamate.agents.coder import build_task
from datamate.agents.facts import verified_facts
from datamate.agents.guideline import parse_rows, render, select

BOOK = """# Rulebook
## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Bán được bao nhiêu tiền, thu về bao nhiêu tiền | là **doanh thu** theo mục này |
| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**) |
## 2. Thời gian
| Cách nói | Nghĩa |
|---|---|
| Tháng M/Y | cả tháng, từ ngày 1 đến hết ngày cuối tháng |
| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59, ca chiều 14:00–21:59 |
## 3. Xếp hạng
| Luật | Mặc định |
|---|---|
| Bán được bao nhiêu <đơn vị> | cộng số lượng quy về đơn vị cơ bản |
## 9. Đáp án
| type | Format | Đúng | Sai |
|---|---|---|---|
| number | JSON number | `1` | `"1"` |
"""
ROWS = parse_rows(BOOK)


def _labels(entries):
    return [e.label for e in entries]


def test_rows_are_the_two_column_table_rows_with_their_section():
    assert _labels(ROWS) == ["Bán được bao nhiêu tiền, thu về bao nhiêu tiền", "Tỉ lệ đơn huỷ trong kỳ", "Tháng M/Y",
                             "Ca sáng, ca chiều", "Bán được bao nhiêu <đơn vị>"]       # headers and the 4-column table skipped
    assert ROWS[1].section.startswith("1.") and "**" not in ROWS[1].text


def test_a_question_selects_its_rows_and_how_to_apply_them_in_this_shop():
    for q in ("Tháng 3/2026 cửa hàng có bao nhiêu % đơn bị huỷ?", "ti le huy don thang 11/2025 la bnhiu %"):
        es = select(q, "2026-05-31", ROWS, have={"cancel_rate"})
        assert _labels(es) == ["Tỉ lệ đơn huỷ trong kỳ"] and es[0].how == "`cancel_rate(period_start, period_end)`"
    es = select("tỉ lệ đơn huỷ tháng 6/2025", "2026-05-31", ROWS, have=set())       # no engine here: the recipe instead
    assert "EVERY order placed in the period" in es[0].how
    # a period definition is left out (the Coder gets the dates); the shift rule says how to compute and stays
    es = select("Tháng 8/2025, doanh thu ca chiều tăng hay giảm bao nhiêu % so với ca sáng vậy?", "2026-05-31", ROWS)
    assert _labels(es) == ["Ca sáng, ca chiều"] and "sales_orders['shift']" in es[0].how
    # "bán được bao nhiêu tiền" is the money row, not "bán được bao nhiêu <đơn vị>"; a unit word is the unit row
    assert _labels(select("Tháng 9/2025 cửa hàng bán được bao nhiêu tiền?", "2026-05-31", ROWS)) == [
        "Bán được bao nhiêu tiền, thu về bao nhiêu tiền"]
    assert _labels(select("quán bán được bao nhiêu ly size M tháng 6", "2026-05-31", ROWS)) == ["Bán được bao nhiêu <đơn vị>"]
    assert select("khách nào mua nhiều nhất", "2026-05-31", ROWS) == []


def test_the_spec_rows_come_first_are_all_kept_and_are_not_repeated():
    spec = ["| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**) |",
            "| Bán được bao nhiêu <đơn vị> | cộng số lượng quy về đơn vị cơ bản |", "a README line the explainer quoted"]
    es = select("tỉ lệ đơn huỷ tháng này, ca sáng", "2026-05-31", ROWS, spec_rules=spec, limit=2)
    assert [e.source for e in es] == ["spec", "spec", "spec"]                 # 3 spec rows > limit 2: all kept, no match added
    assert _labels(es)[:2] == ["Tỉ lệ đơn huỷ trong kỳ", "Bán được bao nhiêu <đơn vị>"] and es[2].text == spec[2]
    es = select("tỉ lệ đơn huỷ tháng này, ca sáng", "2026-05-31", ROWS, spec_rules=spec[:1], limit=3)
    assert _labels(es) == ["Tỉ lệ đơn huỷ trong kỳ", "Ca sáng, ca chiều"]   # the matched duplicate is not shown twice


def test_shop_reminders_come_from_the_data():
    so = pd.DataFrame({"order": ["A"], "date": pd.to_datetime(["2026-05-01"]), "valid": [True], "status_std": ["completed"],
                       "Tổng sau giảm giá": [10.0]})
    so.attrs["header_money"] = ["Tổng sau giảm giá"]
    es = select("doanh thu đơn huỷ tháng 5", "2026-05-31", [], sales_orders=so, measure="money")
    assert [e.source for e in es] == ["shop", "shop"]
    assert "`Tổng sau giảm giá`" in es[0].text and "status_std" in es[1].how
    assert select("doanh thu tháng 5", "2026-05-31", [], sales_orders=so, measure="quantity") == []
    text = render(es)
    assert text.startswith("# RULES FOR THIS QUESTION") and "  -> `sales_orders['revenue']`" in text
    assert render([]) == ""


def test_the_train_rulebook_parses_and_selects():
    book = Path(__file__).resolve().parents[1] / "train" / "rulebook.md"
    if not book.exists():
        return
    rows = parse_rows(book.read_text(encoding="utf-8"))
    assert len(rows) > 40
    es = select("Cho hỏi tháng 3/2026 cửa hàng có bao nhiêu % đơn bị huỷ?", "2026-05-31", rows, have={"cancel_rate"})
    assert es and es[0].label.startswith("Tỉ lệ đơn huỷ") and "cancel_rate" in es[0].how
    assert not any(e.label.startswith("Tháng M") for e in select("doanh thu tháng 6/2025", "2026-05-31", rows))


# ------------------------------------------------------------------ VERIFIED FACTS
SO = pd.DataFrame({"order": ["A", "B", "C", "D", "E"],
                   "date": pd.to_datetime(["2025-08-01 08:00", "2025-08-02 15:00", "2025-08-03 09:00", "2025-08-20 10:00", "2025-09-01 10:00"]),
                   "valid": [True, True, False, True, True], "status_std": ["completed", "completed", "cancelled", "shipping", "completed"],
                   "shift": ["sáng", "chiều", "sáng", "sáng", "sáng"], "Kênh": ["Zalo", "Shopee", "Zalo", "Zalo", "Zalo"]})
IL = pd.DataFrame({"order": ["A", "B", "D"], "date": SO["date"].iloc[[0, 1, 3]].values, "valid": [True, True, True],
                   "key": ["banh a", "banh b", "banh a"]})
VARS = {"sales_orders": SO, "item_lines": IL, "period_start": pd.Timestamp("2025-08-01"),
        "period_end": pd.Timestamp("2025-08-31 23:59:59.999999"), "compare_start": None, "compare_end": None}


def test_verified_facts_count_orders_by_status_shift_entity_and_item():
    spec = SimpleNamespace(entities=[{"value": "Zalo", "column": "Kênh", "role": "other"}, {"value": "Bánh A", "role": "item"}])
    text = verified_facts("Tháng 8/2025 doanh thu ca chiều so với ca sáng", spec, VARS)
    assert text.startswith("# VERIFIED FACTS")
    assert "5 orders, order dates 2025-08-01..2025-09-01" in text
    assert "main period 2025-08-01..2025-08-31: 4 orders placed (completed 2, cancelled 1, shipping 1), 3 valid." in text
    assert "valid orders by `sales_orders['shift']`: sáng 2, chiều 1" in text
    assert "valid orders with `Kênh` == 'Zalo': 2" in text
    assert "valid sales lines of item 'Bánh A' in the main period: 2 (in 2 orders; 2 lines in all the data)" in text
    # no shift line unless the question names a shift ("cà phê" folds to "ca phe": not a shift)
    assert "shift" not in verified_facts("Tháng 8/2025 bán được bao nhiêu ly cà phê", SimpleNamespace(entities=[]), VARS)


def test_verified_facts_say_an_empty_period_is_no_data_and_never_fail():
    empty = {**VARS, "period_start": pd.Timestamp("2026-01-01"), "period_end": pd.Timestamp("2026-01-31 23:59")}
    assert "No order in this period: no data is not 0" in verified_facts("tháng 1/2026", None, empty)
    assert verified_facts("x", None, {}) == ""
    assert verified_facts("x", None, {"sales_orders": "not a frame"}) == ""


def test_the_coder_task_puts_facts_and_rules_right_before_output():
    from datamate.agents.explainer import Spec
    s = Spec(rewritten_question="cancel rate in 2025-08", answer_type="percent", measure="ratio",
             rules=["| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ |"])
    st = SimpleNamespace(brief=s, plan="1. count", engine_ref=None, engine_hint=None, context=None, as_of="2026-05-31",
                         question="tỉ lệ đơn huỷ tháng 8/2025", schema="", ctx=SimpleNamespace(readme=""),
                         facts="# VERIFIED FACTS (counted)\n- x\n\n",
                         guideline=select("tỉ lệ đơn huỷ tháng 8/2025", "2026-05-31", ROWS, spec_rules=s.rules, have={"cancel_rate"}))
    task = build_task(st, "RULEBOOK", ["cancel_rate"])
    assert task.index("# PLAN") < task.index("# VERIFIED FACTS") < task.index("# RULES FOR THIS QUESTION") < task.index("# OUTPUT")
    assert "CALCULATION RULES" not in task and task.count("Tỉ lệ đơn huỷ trong kỳ") == 1
    st.guideline, st.facts = [], ""                                          # switched off: the spec's rows stay where they were
    assert "# CALCULATION RULES" in build_task(st, "RULEBOOK", ["cancel_rate"])



def test_a_pointer_naming_a_column_the_shop_lacks_is_not_shown():
    q = "Tháng 8/2025, doanh thu ca chiều tăng hay giảm bao nhiêu % so với ca sáng vậy?"
    with_shift = select(q, "2026-05-31", ROWS, columns={"order", "valid", "revenue", "shift"})
    assert "sales_orders['shift']" in with_shift[0].how
    no_shift = select(q, "2026-05-31", ROWS, columns={"order", "valid", "revenue"})    # orders without a creation time
    assert _labels(no_shift) == ["Ca sáng, ca chiều"] and no_shift[0].how == ""        # the rule as written, no dead pointer
    es = select("tỉ lệ đơn huỷ tháng 6", "2026-05-31", ROWS, have=set(), columns={"order", "valid"})   # no cancel_rate, no status_std
    assert es[0].how == ""



def test_spec_headings_are_dropped_and_numbered_labels_find_their_row():
    spec = ["## 1. Đơn hàng và doanh thu", "1. Tỉ lệ đơn huỷ trong kỳ: phần trăm đơn huỷ", "## 3. Xếp hạng"]
    es = select("doanh thu tháng 5", "2026-05-31", ROWS, spec_rules=spec, have={"cancel_rate"})
    assert [(e.source, e.label) for e in es] == [("spec", "Tỉ lệ đơn huỷ trong kỳ")] and "cancel_rate" in es[0].how


def test_a_list_answer_form_that_pairs_names_with_amounts_gives_way_to_the_rulebook():
    from datamate.agents.coder import LIST_RULE, coder_brief
    from datamate.agents.explainer import Spec
    from datamate.answer.evaluator import JUDGE_SYSTEM
    bad = ("A list of 5 items, each containing the item name and the total quantity sold in its base unit, sorted by quantity "
           "descending.")                                           # dev5 q_33e8f7506387, 3 of 4 local runs
    _, out = coder_brief(Spec(answer_type="list", k=5, answer_form=bad))
    assert "each containing" not in out and LIST_RULE in out
    for keep in ("A list of 5 item names, sorted in descending order by total quantity sold.",
                 "list of 2 numbers in order: new debt, repaid debt"):
        _, out = coder_brief(Spec(answer_type="list", answer_form=keep))
        assert keep in out and LIST_RULE in out                     # a names-only / numbers-only description stays
    _, out = coder_brief(Spec(answer_type="table", answer_form="columns: customer name and amount owed"))
    assert "amount owed" in out and LIST_RULE not in out            # a table keeps its columns
    assert "never ask to add amounts" in JUDGE_SYSTEM
