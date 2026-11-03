"""Agents harness: Column Selector, Explainer loop (lookups -> spec -> data check -> clarity check), debt functions,
and the whole pipeline with scripted models (no model server)."""
import json
from types import SimpleNamespace

import pandas as pd

from datamate.agents.column_selector import ColumnSelector, check
from datamate.agents.explainer import (
    CHECK_SYSTEM,
    PLAN_SYSTEM,
    TYPE_SYSTEM,
    ExplainerAgent,
    repair_spec,
    run_lookups,
)
from datamate.agents.lookups import DataTools
from datamate.data.loader import Business
from datamate.engines.debt import build_debt
from datamate.engines.functions import build_tools

TABLES = {
    "don": pd.DataFrame({"Mã đơn": ["O1", "O2", "O3"], "Ngày": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]),
                         "Tên hàng": ["Bánh tiramisu", "Cà phê sữa (M)", "Bánh tiramisu"], "Thành tiền": [30.0, 25.0, 30.0]}),
}


class Scripted:
    """Fake model: replies by stage (system prompt) from per-stage queues; records every prompt."""

    def __init__(self, **queues):
        self.queues, self.seen = {k: list(v) for k, v in queues.items()}, []

    def generate(self, messages):
        system, user = messages[0]["content"][0]["text"], messages[1]["content"][0]["text"]
        stage = "check" if system == CHECK_SYSTEM else "type" if system == TYPE_SYSTEM else "plan" if system == PLAN_SYSTEM else "select" if "BỘ CHỌN CỘT" in system else "explain"
        self.seen.append((stage, user))
        q = self.queues.get(stage) or ["{}"]
        out = q.pop(0) if len(q) > 1 else q[0]
        return SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))


def _st(question="tiramisu thang 5 ban dc bn cai"):
    ctx = SimpleNamespace(tables=TABLES, descriptors={"don": "Table `don` (3 rows)"}, readme="", schema="Table `don`", items=None)
    return SimpleNamespace(q={"question": question, "as_of_date": "2026-05-31"}, question=question, as_of="2026-05-31",
                           ctx=ctx, log={}, schema="Table `don`", plan="")


SPEC = {"rewritten_question": "Tháng 5/2026 (2026-05-01..2026-05-31) bán được bao nhiêu cái 'Bánh tiramisu' (đơn hợp lệ)?",
        "answer_type": "number", "answer_form": "một số, đơn vị cái", "k": None,
        "periods": [{"label": "tháng 5/2026", "start": "2026-05-01", "end": "2026-05-31", "role": "main"}],
        "entities": [{"value": "Bánh tiramisu", "table": "don", "column": "Tên hàng", "role": "item"}],
        "tables": {"don": ["Tên hàng", "Ngày"]}, "rules": [], "answerable": True}


def test_selector_reports_unknown_names_and_keeps_valid_ones():
    sel, errs = check({"tables": {"don": ["Ngày", "Ngay ban"], "orders": ["x"]}}, TABLES)
    assert sel == {"don": ["Ngày"]}
    assert any("Ngay ban" in e for e in errs) and any("orders" in e for e in errs)
    st = _st()
    ColumnSelector(Scripted(select=["not json"])).run(st)          # nothing usable -> every table, nothing lost
    assert st.selection == {"don": list(TABLES["don"].columns)}


def test_spec_is_checked_against_the_data():
    bad = {**SPEC, "entities": [{"value": "banh tiramisu", "table": "don", "column": "Tên hàng"}], "answer_type": "count"}
    s, errs, _, _ = repair_spec(bad, TABLES, "2026-05-31")
    assert any("answer_type" in e for e in errs) and any("is not in the data" in e for e in errs)
    assert s.answer_type is None and not s.entities                 # the wrong fields are dropped, the rest is kept
    s, errs, warns, _ = repair_spec({**SPEC, "periods": [{"label": "x", "start": "2026-06-01", "end": "2026-06-30", "role": "main"}]},
                                    TABLES, "2026-05-31")
    assert s is not None and not errs and warns                    # future period: a warning, not an error


def test_explainer_looks_up_then_passes_the_clarity_check():
    m = Scripted(explain=[{"lookups": [{"tool": "find_values", "args": {"text": "tiramisu"}}], "spec": None}, {"lookups": [], "spec": SPEC}],
                 check=[{"clear": True, "issues": []}], plan=["1. lọc ... 2. cộng ..."])
    st = _st()
    s = ExplainerAgent(m, m, m, "RULEBOOK", pd.DataFrame()).run(st)
    assert s.status == "clear" and st.answer_type == "number"
    assert str(st.period.main.start) == "2026-05-01" and st.brief.rewritten_question.startswith("Tháng 5/2026")
    second_turn = [u for stage, u in m.seen if stage == "explain"][1]
    assert "value='Bánh tiramisu'" in second_turn                  # the lookup result reached the model
    assert st.plan.startswith("1.")


def test_clarity_issues_go_back_to_the_explainer():
    vague = {**SPEC, "rewritten_question": "bán được bao nhiêu tiramisu"}
    m = Scripted(explain=[{"spec": vague}, {"spec": SPEC}], check=[{"clear": False, "issues": [{"field": "rewritten_question", "quote": "bn cai", "problem": "đại lượng chưa rõ: số cái hay tiền?", "fix": "ghi rõ số cái"}]},
                                                                    {"clear": True, "issues": []}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.status == "clear" and s.rewritten_question == SPEC["rewritten_question"]
    assert "đại lượng chưa rõ" in [u for stage, u in m.seen if stage == "explain"][1]


def test_unfinished_spec_still_reaches_the_coder():
    m = Scripted(explain=["garbage"])                              # never valid: fallback, not an abstain
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_calls=2).run(st)
    assert s.status == "fallback" and s.rewritten_question == st.question and st.period.main is None


def _debt_biz():
    orders = pd.DataFrame({
        "Mã đơn": ["O1", "O2", "O3", "O4", "O5"], "Ngày": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-20", "2026-05-25"]),
        "Mã KH": ["KH1", "KH2", "KH1", None, "KH2"], "Phương thức": ["Ghi nợ", "Ghi nợ", "Ghi nợ", "Ghi nợ", "Thu hộ (COD)"],
        "status_std": ["completed", "returned", "cancelled", "completed", "completed"], "Tổng đơn": [100.0, 200.0, 300.0, 50.0, 80.0]})
    pays = pd.DataFrame({"Mã phiếu": ["P1", "P2", "P3", "P4"], "Mã đơn": ["O1", "O1", "O2", "O4"],
                         "Ngày thu": pd.to_datetime(["2026-05-01", "2026-05-10", "2026-05-15", "2026-05-20"]), "Tiền": [40.0, 30.0, 50.0, 10.0]})
    kh = pd.DataFrame({"Mã khách hàng": [f"KH{i}" for i in range(1, 31)], "Tên khách hàng": [f"Khách Số {chr(65 + i % 26)}{chr(65 + i // 26)} Hà" for i in range(30)]})
    readme = "## Quy ước\n- **Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **Tiền** là số tiền trả lần đó; **Tổng đơn** là số khách phải trả cho cả đơn."
    return Business("b", readme, {"don_hang": orders, "thanh_toan": pays, "khach_hang": kh})


def test_debt_follows_rulebook_section_4():
    v = build_debt(_debt_biz())
    t = build_tools(None, None, v)
    # end of 31/05: O1 100-70=30, O2 (returned, still debt) 200-50=150, O3 cancelled, O4 walk-in 50-10=40, O5 COD never debt
    assert t["debt_total"]("2026-05-31") == 220
    assert [n for n, _ in t["debt_by_customer"]("2026-05-31")] == [v.orders.loc[v.orders["cust"] == "KH2", "name"].iloc[0],
                                                                  v.orders.loc[v.orders["cust"] == "KH1", "name"].iloc[0]]
    assert t["debt_customers"]("2026-05-31") == 2                  # walk-in O4 has no row
    assert t["debt_aging"]("2026-05-31") == [0, 40, 180]           # ages 30, 29, 11 days
    assert t["debt_repaid"]("2026-05-01", "2026-05-31") == 80      # P1 and P4 were paid on the purchase day: not repayments
    assert t["debt_new"]("2026-05-01", "2026-05-31") == 100 - 40 + 200 + 50 - 10


def test_the_payment_method_column_is_found_by_the_readme_or_a_shared_name_pattern():
    """B6: one payment-column definition for status and debt; `Thanh toán bằng` (dev_008's name) and a README-declared column count."""
    base = _debt_biz()
    for col, readme_line in (("Thanh toán bằng", ""), ("Kiểu", "\n## Cột trong `don_hang.csv`\n- **Mã đơn**: mã đơn\n- **Kiểu**: hình thức thanh toán của đơn\n"
                                                          "- **Ngày**: ngày bán\n- **Tổng đơn**: tổng tiền\n- **Mã KH**: mã khách\n")):
        orders = base.tables["don_hang"].rename(columns={"Phương thức": col})
        v = build_debt(Business("b", base.readme + readme_line, {**base.tables, "don_hang": orders}))
        assert v is not None and build_tools(None, None, v)["debt_total"]("2026-05-31") == 220, col     # same as with "Phương thức"
    orders = base.tables["don_hang"].rename(columns={"Phương thức": "Kiểu"})                             # no README line, no pattern
    v = build_debt(Business("b", base.readme, {**base.tables, "don_hang": orders}))
    assert build_tools(None, None, v)["debt_total"]("2026-05-31") == 300      # unseen: the COD order O5 (80) is counted as owed (§4 says never)


def test_repeated_lookups_are_not_rerun_and_a_spec_is_demanded():
    lk = {"lookups": [{"tool": "find_values", "args": {"text": "tiramisu"}}], "spec": None}
    m = Scripted(explain=[lk, lk, {"lookups": [], "spec": SPEC}], check=[{"clear": True, "issues": []}])
    st = _st()
    ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    second = [u for stage, u in m.seen if stage == "explain"][1]
    assert "MUST contain a spec" in second
    third = [u for stage, u in m.seen if stage == "explain"][2]
    assert "ALREADY LOOKED UP" in third


def test_a_valid_spec_sent_with_lookups_is_still_checked():
    m = Scripted(explain=[{"lookups": [{"tool": "date_info", "args": {"day": "2026-05-30"}}], "spec": SPEC}],
                 check=[{"clear": True, "issues": []}])
    assert ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st()).status == "clear"


def test_entity_spelled_differently_is_snapped_to_the_data():
    d = {**SPEC, "entities": [{"value": "BANH TIRAMISU", "table": "don", "column": "Tên hàng", "role": "item"}]}
    s, errs, _, notes = repair_spec(d, TABLES, "2026-05-31", DataTools(TABLES))
    assert not errs and s.entities[0]["value"] == "Bánh tiramisu" and "Bánh tiramisu" in notes[0]


def test_wrong_fields_are_dropped_not_the_whole_spec():
    d = {**SPEC, "entities": [{"value": "tháng 5/2025", "table": "don", "column": "Ngày"}, {"value": "Phở bò", "table": "don", "column": "Tên hàng"}],
         "tables": {"don": ["Tên hàng", "Ngay"], "orders": ["x"]}}
    s, errs, _, _ = repair_spec(d, TABLES, "2026-05-31", DataTools(TABLES))
    assert s.answer_type == "number" and s.periods == SPEC["periods"] and s.entities == []
    assert s.tables == {"don": ["Tên hàng"]}
    assert any("periods" in e for e in errs)                       # typed feedback: a date is not an entity
    assert any("Phở bò" in e for e in errs)


def test_partial_spec_beats_the_raw_question_when_the_budget_ends():
    bad = {**SPEC, "entities": [{"value": "Phở bò", "table": "don", "column": "Tên hàng"}]}
    m = Scripted(explain=[{"spec": bad}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_calls=2).run(_st())
    # the same entity error twice is settled (field dropped, noted), so the partial spec is then checked
    assert s.status == "clear" and s.answer_type == "number" and str(s.periods[0]["start"]) == "2026-05-01"
    assert not s.entities and any("'Phở bò'" in a for a in s.assumptions)


def test_malformed_lookup_gets_the_correct_form():
    out = run_lookups(DataTools(TABLES), [{"tool": "find_values"}, {"tool": "find_values", "args": {"txt": "x"}}])
    assert "missing args" in out[0] and '"text"' in out[0] and '"text"' in out[1]


def test_independent_type_reading_is_fed_back_once():
    listy = {**SPEC, "answer_type": "list"}
    m = Scripted(explain=[{"spec": listy}, {"spec": SPEC}], type=[{"answer_type": "number", "quote": "bn cai"}],
                 check=[{"clear": True, "issues": []}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.status == "clear" and s.answer_type == "number"
    assert "an independent reader chose 'number'" in [u for stage, u in m.seen if stage == "explain"][1]


def test_clarity_issues_without_evidence_are_dropped():
    invented = {"field": "filters", "quote": "khách VIP", "problem": "lọc theo khách", "fix": "bỏ"}
    m = Scripted(explain=[{"spec": SPEC}], check=[{"clear": False, "issues": [invented, "plain text issue"]}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.status == "clear"                                      # "khách VIP" is not in the question
    assert len(st.log["explainer"]["turns"][0]["clarity"]["dropped"]) == 2


def test_blind_reading_with_an_invented_quote_is_ignored():
    m = Scripted(explain=[{"spec": SPEC}], type=[{"answer_type": "percent", "measure": "ratio", "quote": "tỉ lệ bán"}],
                 check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    assert s.status == "clear" and s.answer_type == "number"         # "tỉ lệ bán" is not in the question


def test_measure_disagreement_goes_back_to_the_explainer():
    money = {**SPEC, "measure": "money", "rewritten_question": "Doanh thu 'Bánh tiramisu' tháng 5/2026?"}
    m = Scripted(explain=[{"spec": money}, {"spec": {**SPEC, "measure": "quantity"}}],
                 type=[{"answer_type": "number", "measure": "quantity", "quote": "bn cai"}], check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    assert s.measure == "quantity" and s.status == "clear"
    assert "measuring 'quantity'" in [u for stage, u in m.seen if stage == "explain"][1]


def _orders_biz():
    names = [f"Khách Số {chr(65 + i % 26)}{chr(65 + i // 26)} Hà" for i in range(30)]
    hdr = pd.DataFrame({"Mã đơn": [f"O{i}" for i in range(1, 7)],
                        "Ngày": pd.to_datetime(["2026-05-01"] * 3 + ["2026-05-02"] * 3),
                        "Mã KH": ["KH1", "KH2", "KH1", None, "KH2", "KH3"], "Kênh": ["store", "zalo", "store", "store", "store", "store"],
                        "Tổng sau giảm giá": [999.0] * 6,
                        "status_std": ["completed", "completed", "cancelled", "completed", "completed", "returned"]})
    hdr["valid_order"] = hdr["status_std"].isin(["completed", "shipping"])
    lines = pd.DataFrame({"Mã đơn": ["O1", "O1", "O2", "O3", "O4", "O5", "O6", "O5"],
                          "Tên hàng": ["A", "B", "A", "A", "B", "A", "B", "Phí ship"], "Thành tiền": [10.0, 5.0, 40.0, 70.0, 8.0, 12.0, 90.0, 3.0]})
    kh = pd.DataFrame({"Mã khách hàng": [f"KH{i}" for i in range(1, 31)], "Tên khách hàng": names})
    return Business("b", "", {"hoa_don": hdr, "hoa_don_chi_tiet": lines, "khach_hang": kh}), names


def test_sales_orders_carry_rulebook_revenue_and_rank_customers():
    from datamate.engines.orders import build_orders, top_customers
    from datamate.engines.sales import build_sales
    biz, names = _orders_biz()
    o = build_orders(biz, build_sales(biz)).set_index("order")
    assert o.loc["O1", "revenue"] == 15 and o.loc["O5", "revenue"] == 12      # line sum, delivery fee is not revenue
    assert o.loc["O3", "revenue"] == 0 and not o.loc["O3", "valid"]           # cancelled
    assert o.loc["O6", "revenue"] == 0                                        # returned
    assert o.loc["O4", "walk_in"] and o.loc["O2", "Kênh"] == "zalo"
    rows = top_customers(o.reset_index(), None, "2026-05-01", "2026-05-02")
    assert rows == [(names[1], 52), (names[0], 15)]                          # KH2 40+12, KH1 15; walk-in O4 not ranked


def test_summing_an_order_header_total_is_refused_once():
    import pytest

    from datamate.sandbox.guard import GuardedExecutor
    ran = []
    ex = GuardedExecutor(lambda code: ran.append(code) or SimpleNamespace(logs="", output=None))
    ex.order_money = {"Tổng sau giảm giá"}
    code = "x = tables['hd']['Tổng sau giảm giá'].sum()\nprint(x)"
    with pytest.raises(ValueError, match="sales_orders"):
        ex(code)
    with pytest.raises(ValueError, match="sales_orders"):
        ex(code)
    ex(code)                                                          # third time it runs as written
    ex2 = GuardedExecutor(lambda code: ran.append(code) or SimpleNamespace(logs="", output=None))
    ex2.order_money = {"Tổng sau giảm giá"}
    ex2("x = sales_orders[sales_orders['valid']]['revenue'].sum()")  # the rulebook revenue is fine
    assert len(ran) == 2


def test_a_customer_the_question_never_names_is_dropped():
    tables = {"don": TABLES["don"].assign(**{"Khách": ["Bạch Ngân Hằng", "Lê Văn An", "Lê Văn An"]})}
    bad = {**SPEC, "entities": [{"value": "Bạch Ngân Hằng", "table": "don", "column": "Khách", "role": "customer"}]}
    s, errs, _, _ = repair_spec(bad, tables, "2026-05-31", DataTools(tables), "bua qua cua hang co bn don")
    assert s.entities == [] and "does not name" in errs[0]
    ok = {**SPEC, "entities": [{"value": "Lê Văn An", "table": "don", "column": "Khách", "role": "customer"}]}
    s, errs, _, _ = repair_spec(ok, tables, "2026-05-31", DataTools(tables), "khach le van an mua may don")
    assert len(s.entities) == 1 and not errs


def test_two_readings_of_a_count_turn_a_list_spec_into_a_number():
    listy = {**SPEC, "answer_type": "list", "measure": "count"}
    m = Scripted(explain=[{"spec": listy}], type=[{"answer_type": "number", "measure": "count", "quote": "ban dc bn cai"}],
                 check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_calls=1).run(_st())
    assert s.answer_type == "number"


def test_one_measure_reading_alone_does_not_flip_the_spec():
    qty = {**SPEC, "measure": "quantity"}
    m = Scripted(explain=[{"spec": qty}], check=[{"clear": True, "issues": []}],
                 type=[{"answer_type": "number", "measure": "money", "quote": "ban dc"},
                       {"answer_type": "number", "measure": "quantity", "quote": "bn cai"}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.measure == "quantity" and s.status == "clear"            # the second reading disagrees: not fed back
    assert st.log["explainer_blind_type2"]["measure"] == "quantity"


def test_rewrite_naming_a_dropped_person_falls_back_to_the_question():
    tables = {"don": TABLES["don"].assign(**{"Khách": ["Bạch Ngân Hằng", "Lê Văn An", "Lê Văn An"]})}
    bad = {**SPEC, "rewritten_question": "Hôm qua khách 'Bạch Ngân Hằng' có bao nhiêu đơn?",
           "entities": [{"value": "Bạch Ngân Hằng", "table": "don", "column": "Khách", "role": "customer"}]}
    s, _, _, _ = repair_spec(bad, tables, "2026-05-31", DataTools(tables), "bua qua cua hang co bn don")
    assert s.rewritten_question == "bua qua cua hang co bn don"


def test_a_month_answer_read_as_a_date_is_typed_date():
    stringy = {**SPEC, "answer_type": "string", "measure": "money"}
    m = Scripted(explain=[{"spec": stringy}], check=[{"clear": True, "issues": []}],
                 type=[{"answer_type": "date", "measure": "date", "quote": "bn cai"}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_checks=1).run(_st())
    assert s.answer_type == "date"
    m = Scripted(explain=[{"spec": stringy}], check=[{"clear": True, "issues": []}],
                 type=[{"answer_type": "string", "measure": "date", "quote": "bn cai"}])
    assert ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_checks=1).run(_st()).answer_type == "date"


def test_placeholder_number_is_flagged():
    from datamate.answer.interpreter import interpret
    assert "sentinel" in interpret(999999999, "number").issues
    assert "sentinel" not in interpret(1999999, "number").issues


def test_a_word_read_as_a_missing_entity_does_not_stay_in_the_rewrite():
    bad = {**SPEC, "rewritten_question": "Số đợt giao của xe biển số 'ddax' tháng 8?",
           "entities": [{"value": "ddax", "table": "don", "column": "Tên hàng", "role": "other"}]}
    s, errs, _, _ = repair_spec(bad, TABLES, "2026-05-31", DataTools(TABLES), "thang 8 xe ddax giao may dot")
    assert s.entities == [] and s.rewritten_question == "thang 8 xe ddax giao may dot"


def test_count_agreed_by_both_readings_is_a_number_even_if_listed():
    listy = {**SPEC, "answer_type": "list", "measure": "count"}
    m = Scripted(explain=[{"spec": listy}], check=[{"clear": True, "issues": []}],
                 type=[{"answer_type": "list", "measure": "count", "quote": "bn cai"}])
    assert ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_checks=1).run(_st()).answer_type == "number"


def test_item_lines_carry_product_attributes():
    from datamate.engines.items import with_item_attrs
    lines = pd.DataFrame({"code": ["P1", "P2", "P1"], "amount": [10.0, 5.0, 7.0]})
    prod = pd.DataFrame({"Mã hàng": ["P1", "P2", "P3"], "Nhóm hàng": ["bia", "sữa", "gạo"]})
    drugs = pd.DataFrame({"Mã hàng": ["P1", "P2"], "Thuốc kê đơn": [True, False]})
    o = with_item_attrs(lines, {"sp": prod, "thuoc": drugs, "other": pd.DataFrame({"x": [1, 1]})})
    assert o["Nhóm hàng"].tolist() == ["bia", "sữa", "bia"] and o["Thuốc kê đơn"].tolist() == [True, False, True]
    assert o[o["Nhóm hàng"] == "bia"]["amount"].sum() == 17


def test_shift_comes_from_the_column_with_times():
    from datamate.engines.orders import _with_shift
    o = pd.DataFrame({"date": pd.to_datetime(["2025-03-01"] * 3), "Thời gian": pd.to_datetime(["2025-03-01"] * 3),
                      "Thời gian tạo": pd.to_datetime(["2025-03-01 07:10", "2025-03-01 14:00", "2025-03-01 23:30"])})
    o = _with_shift(o)
    assert o.attrs["shift_from"] == "Thời gian tạo" and o["shift"].tolist()[:2] == ["sáng", "chiều"] and pd.isna(o["shift"].iloc[2])


def test_shift_reads_times_kept_as_text():
    from datamate.engines.orders import _with_shift
    o = pd.DataFrame({"date": pd.to_datetime(["2025-03-01"] * 2), "Giờ": ["2025-03-01 06:13:20", "2025-03-01 15:00:00"]})
    o = _with_shift(o)
    assert o.attrs["shift_from"] == "Giờ" and o["shift"].tolist() == ["sáng", "chiều"]


def test_an_entity_the_question_names_is_not_dropped_later():
    # r6 "nhom bia ...": the checker said 'bia' was not in the question; the next spec dropped it
    dropped = {**SPEC, "entities": [], "rewritten_question": "Tháng 5/2026 bán được bao nhiêu cái?"}
    m = Scripted(explain=[{"spec": SPEC}, {"spec": dropped}],
                 check=[{"clear": False, "issues": [{"field": "entities", "quote": "thang 5", "problem": "thừa", "fix": "bỏ"}]},
                        {"clear": True, "issues": []}])
    st = _st("banh tiramisu thang 5 ban dc bn cai")
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert [e["value"] for e in s.entities] == ["Bánh tiramisu"]
    assert s.rewritten_question == st.question          # the rewrite lost the item: the original question is used


def test_a_repeated_spec_error_is_settled_and_the_spec_checked():
    # r6 "xe ddax giao": the same "not in the data" error 5 turns running left the spec unchecked
    ghost = {**SPEC, "entities": SPEC["entities"] + [{"value": "ddax", "table": "don", "column": "Tên hàng", "role": "other"}]}
    m = Scripted(explain=[{"spec": ghost}, {"spec": ghost}], check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st("tiramisu ddax thang 5 ban dc bn cai"))
    assert s.status == "clear" and [e["value"] for e in s.entities] == ["Bánh tiramisu"]
    assert any("'ddax' is not a value in the data" in a for a in s.assumptions)


def test_two_agreeing_readings_override_a_kept_measure():
    money = {**SPEC, "measure": "money", "rewritten_question": "Doanh thu tiramisu tháng 5?"}
    reading = {"answer_type": "number", "quote": "bn cai", "measure": "quantity"}
    m = Scripted(explain=[{"spec": money}], type=[reading], check=[{"clear": True, "issues": []}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.measure == "quantity" and s.rewritten_question == st.question


def test_missing_column_error_points_to_sales_orders():
    from datamate.sandbox.guard import GuardedExecutor
    ex = GuardedExecutor(None)
    ex.tables = {"HoaDon": pd.DataFrame({"Mã": ["A"], "Tổng": [1]})}
    ex.frames = {"sales_orders": pd.DataFrame({"order": ["A"], "shift": ["sáng"], "revenue": [1]})}
    msg = ex.explain_error("Could not index ...", "df = tables['HoaDon']\nm = df['shift'] == 'sáng'")
    assert "exists only in the variable `sales_orders`" in msg and "'shift'" in msg


def test_a_count_in_any_shape_is_one_number():
    # r7 "mấy mặt hàng ế": 2, {'count': 2, 'item_names': [...]} and a list of the 2 names
    from datamate.answer.interpreter import as_count
    assert [as_count(r) for r in (2, {"count": 2, "item_names": ["A", "B"]}, ["A", "B"])] == [2, 2, 2]
    assert as_count({"a": 1, "b": 2}) == {"a": 1, "b": 2}      # two numbers: not a count, left alone


def test_only_missing_entities_are_settled_when_repeated():
    # a repeated period error (start after end) is not settled: it goes back to the Explainer
    bad = {**SPEC, "periods": [{**SPEC["periods"][0], "start": "2026-06-01"}]}
    m = Scripted(explain=[{"spec": bad}, {"spec": bad}, {"spec": SPEC}], check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    assert s.status == "clear" and s.periods[0]["start"] == "2026-05-01"
    assert not any("after end" in a for a in s.assumptions)


def test_equal_periods_without_a_main_one_are_not_an_error():
    # public run "tháng nào cao nhất trong 3 tháng": all compare, the error kept the spec from the check
    s, errs, _, notes = repair_spec({**SPEC, "periods": [{**SPEC["periods"][0], "role": "compare"}]}, TABLES, "2026-05-31")
    assert not errs and s.periods[0]["role"] == "main" and any("role 'main'" in n for n in notes)


def test_month_named_in_words_is_a_date():
    from datamate.answer.interpreter import month_answer
    P = [{"start": "2026-02-01"}, {"start": "2026-03-01"}]
    assert month_answer("Tháng 2", P) == "2026-02" and month_answer("tháng 7/2025", P) == "2025-07"
    assert month_answer("tháng 5 năm 2025", []) == "2025-05" and month_answer("Th9/25", []) == "2025-09"
    assert month_answer("Tháng 12", P) is None              # no period in December: the year is unknown
    assert month_answer("Cà phê sữa (M)", P) is None and month_answer("12", P) is None


def test_top_day_outside_the_asked_period_is_flagged():
    from datamate.answer.interpreter import in_periods
    P = [{"start": "2025-09-01", "end": "2025-09-30"}]
    assert not in_periods("2025-08-12", P) and in_periods("2025-09-12", P) and in_periods("2025-09", P)
    assert in_periods("2025-08-12", [])


def test_spec_without_a_check_still_gets_the_independent_reading():
    # public run: "chiếm mấy % tổng đơn" with no usable spec came out as a number
    m = Scripted(explain=["not json"], type=[{"answer_type": "percent", "quote": "bn cai", "measure": "ratio"}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_calls=1).run(st)
    assert s.status == "fallback" and s.answer_type == "percent" and st.log["explainer_blind_type"]["late"]


def test_count_readings_beat_a_list_of_names():
    # public run "có mấy mặt hàng nằm trong ít nhất 122 đơn" answered with the names
    names = {**SPEC, "answer_type": "list", "measure": "name", "rewritten_question": "Liệt kê các món tháng 5?"}
    m = Scripted(explain=[{"spec": names}], type=[{"answer_type": "number", "quote": "bn cai", "measure": "count"}],
                 check=[{"clear": True, "issues": []}])
    st = _st()
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(st)
    assert s.measure == "count" and s.answer_type == "number" and s.rewritten_question == st.question


def test_coder_task_is_framed_context_task_inputs_output():
    from datamate.agents.coder import build_task
    from datamate.agents.explainer import Spec
    spec = Spec(rewritten_question="Doanh thu 3 tháng 10, 11, 12/2025, tháng nào cao nhất?", answer_type="date", measure="money",
                periods=[{"label": "T10", "start": "2025-10-01", "end": "2025-10-31", "role": "main"},
                         {"label": "T11", "start": "2025-11-01", "end": "2025-11-30", "role": "compare"},
                         {"label": "T12", "start": "2025-12-01", "end": "2025-12-31", "role": "compare"}],
                entities=[{"value": "Bánh tiramisu", "table": "don", "column": "Tên hàng", "role": "item"}],
                assumptions=["harness: answer_type string -> date", "doanh thu theo rulebook §1"], status="unchecked",
                open_issues=["periods: ?"])
    st = SimpleNamespace(ctx=SimpleNamespace(readme="README"), schema="S", as_of="2026-09-10", question="q", brief=spec,
                         plan="1. lọc")
    t = build_task(st, "RB", ["sales_metrics"])
    order = ["# CONTEXT", "## RULEBOOK", "## AVAILABLE FUNCTIONS", "## SCHEMA", "## QUESTION", "# TASK", "# RESOLVED INPUTS",
             "# NOTES", "# PLAN", "# OUTPUT"]
    assert [t.index(h) for h in order] == sorted(t.index(h) for h in order)       # sections in framework order
    assert "= variable `period_start .. period_end`" in t and "= variable `compare_start .. compare_end`" in t
    assert t.count("no variable of its own") == 1                                     # the third month has no variable
    assert "- System: answer_type string -> date" in t and "- Explainer's assumption: doanh thu" in t
    assert "did NOT pass every check: periods: ?" in t
    assert "'YYYY-MM'" in t[t.index("# OUTPUT"):] and "Measure: money" in t


def test_checks_run_when_only_a_column_was_dropped():
    # biz_109 q_00a6e23cbacf: a dropped column was an error all loop long and the spec was never read by the checks
    spec = {**SPEC, "tables": {"don": ["Tên hàng", "Cột không có"]}}
    m = Scripted(explain=[{"spec": spec}], check=[{"clear": True, "issues": []}])
    s = ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    assert [k for k, _ in m.seen].count("check") == 1                 # the clarity check ran on the first spec
    assert [k for k, _ in m.seen].count("explain") == 1               # and the spec was accepted without another turn
    assert s.status == "clear" and s.tables == {"don": ["Tên hàng"]}
    assert any("a wrong field was dropped" in a for a in s.assumptions)


def test_checker_issues_on_a_spec_with_a_dropped_column_go_back_with_the_error():
    spec = {**SPEC, "tables": {"don": ["Tên hàng", "Cột không có"]}}
    issue = {"clear": False, "issues": [{"field": "periods", "quote": "thang 5", "problem": "x", "fix": "y"}]}
    m = Scripted(explain=[{"spec": spec}, {"spec": SPEC}], check=[issue, {"clear": True, "issues": []}])
    ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    second = [u for k, u in m.seen if k == "explain"][1]
    assert "SPEC ERRORS" in second and "THE CHECKER STILL SEES PROBLEMS" in second


def test_a_real_error_does_not_spend_a_check():
    bad = {**SPEC, "periods": [{**SPEC["periods"][0], "start": "2026-06-01"}]}
    m = Scripted(explain=[{"spec": bad}, {"spec": SPEC}], check=[{"clear": True, "issues": []}])
    ExplainerAgent(m, m, m, "R", pd.DataFrame()).run(_st())
    assert [k for k, _ in m.seen].count("check") == 1                 # only on the repaired spec


def test_last_turn_without_a_spec_gets_one_forced_call():
    # q_00a6e23cbacf: 5 turns, the last one lookups again; the error never got a second chance
    spec = {**SPEC, "tables": {"don": ["Tên hàng", "Cột không có"]}}
    lookups = {"lookups": [{"tool": "describe", "args": {"table": "don"}}]}
    m = Scripted(explain=[lookups, lookups, lookups, {"spec": spec}, lookups, {"spec": SPEC}],
                 check=[{"clear": False, "issues": [{"field": "periods", "quote": "thang 5", "problem": "x", "fix": "y"}]}])
    agent = ExplainerAgent(m, m, m, "R", pd.DataFrame(), max_calls=5, max_checks=1)
    s = agent.run(_st())
    explains = [u for k, u in m.seen if k == "explain"]
    assert len(explains) == 6 and "FINAL TURN: lookups are closed" in explains[-1]
    assert s.tables == {"don": ["Tên hàng", "Ngày"]}                  # the forced, fully valid spec replaced the one with the error
    assert s.status == "unchecked"
