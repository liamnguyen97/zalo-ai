"""Stage-level behaviour added with the diagram refactor (no model calls)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from datamate.answer.interpreter import interpret
from datamate.data.readme import roles_for_tables
from datamate.sandbox.guard import PROTECTED, GuardedExecutor


class _Inner:
    def __call__(self, code):
        return "ran"


# ---------------------------------------------------------------- interpreter
@pytest.mark.parametrize("raw, expected, kind, value", [
    (pd.Timestamp("2026-04-29"), "date", "date", pd.Timestamp("2026-04-29")),
    ("2026-04-29 00:00:00", "date", "date", "2026-04-29 00:00:00"),
    ({"type": "percent", "answer": -39.96}, None, "percent", -39.96),
    (12.5, "percent", "percent", 12.5),
    (48250000, "number", "number", 48250000),
    (["Cà phê sữa"], "string", "string", "Cà phê sữa"),
    ("1234", "number", "number", 1234.0),
    ("Có", "boolean", "boolean", True),
])
def test_interpreter_types_with_expected(raw, expected, kind, value):
    it = interpret(raw, expected)
    assert (it.kind, it.raw, it.issues) == (kind, value, [])


@pytest.mark.parametrize("raw, expected, issue", [
    ([3, ["Nước giặt", "Sữa tắm"]], "number", "type"),          # "có mấy món" answered with a list
    ([], "list", "empty"),
    (float("nan"), "number", "empty"),
    (["kl", "Khách lẻ", "lẻ"], "list", "walk_in"),               # walk-ins ranked as customers
    ("khách lẻ", "string", "walk_in"),
    ({"columns": ["a", "b"], "rows": [["0-7", 1]]}, "list", "type"),
])
def test_interpreter_flags_doubtful_answers(raw, expected, issue):
    assert issue in interpret(raw, expected).issues


def test_interpreter_unanswerable_passthrough():
    assert interpret("UNANSWERABLE: không có giá vốn", "number").kind == "unanswerable"


# ---------------------------------------------------------------- README roles by columns
def test_readme_sections_match_tables_by_columns():
    readme = ("## Cột trong `BanHang_T<tháng>_<năm>.xlsx`, sheet `DonHang_ChiTiet`\n- **Mã đơn hàng**: mã đơn\n"
              "- **Tổng tiền**: thành tiền sau giảm\n- **TT đơn**: trạng thái đơn\n- **Ngày đặt hàng**: ngày bán\n"
              "## Cột trong `DanhSachQuyDoi.xlsx`\n- **Tên sản phẩm**: tên hàng\n- **Tỉ lệ quy đổi**: hệ số quy đổi\n")
    roles = roles_for_tables(readme, {"BanHang": ["Mã đơn hàng", "Ngày đặt hàng", "Tổng tiền", "TT đơn", "Tên sản phẩm"],
                                      "Khac": ["x", "y"]})
    assert roles == {"BanHang": {"order_id": "Mã đơn hàng", "line_amount": "Tổng tiền", "order_status": "TT đơn", "order_date": "Ngày đặt hàng"}}


# ---------------------------------------------------------------- runner guard
def test_period_vars_protected_only_when_given():
    g = GuardedExecutor(_Inner(), PROTECTED | {"period_end"})
    assert g("period_end = pd.Timestamp('2025-06-08')") == "ran" and g.dropped == 1  # resolved: dropped, not refused
    with pytest.raises(ValueError):
        g("for period_end in [1]: pass")
    assert g("period_start = pd.Timestamp('2025-01-20')") == "ran"  # unresolved (e.g. Tết): the coder sets it


# ---------------------------------------------------------------- loader: one table split over workbooks
def test_year_files_with_different_sheet_layouts_merge(tmp_path):
    from datamate.data.loader import load_business

    d = tmp_path / "biz"; d.mkdir()
    head = ["Ma don", "Ngày", "Tiền"]
    with pd.ExcelWriter(d / "BanHang_TongHop_2025.xlsx") as w:
        pd.DataFrame([head, ["A1", "01/12/2025", 10], ["A2", "02/12/2025", 5]]).to_excel(w, sheet_name="BanHang", header=False, index=False)
        pd.DataFrame([["Mã", "Tên"], ["SP1", "Cát"], ["SP2", "Đá"]]).to_excel(w, sheet_name="HangHoa", header=False, index=False)
    with pd.ExcelWriter(d / "BanHang_TongHop_2026.xlsx") as w:
        pd.DataFrame([head, ["B1", "03/01/2026", 7]]).to_excel(w, sheet_name="BanHang", header=False, index=False)
    tables = load_business(d).tables
    sales = [t for t, df in tables.items() if "Tiền" in df.columns]
    assert len(sales) == 1 and len(tables[sales[0]]) == 3
    assert any("HangHoa" in t for t in tables)  # the other sheet stays its own table


def test_period_redefinitions_are_dropped_not_refused():
    seen = []
    g = GuardedExecutor(lambda code: seen.append(code) or "ran", PROTECTED | {"period_start", "period_end"})
    assert g("import pandas as pd\nperiod_start = pd.Timestamp('2026-01-01')\nperiod_end = pd.Timestamp('2026-01-31')\nx = 1") == "ran"
    assert "period_start" not in seen[-1] and "x = 1" in seen[-1] and g.dropped == 2
    with pytest.raises(ValueError):  # nested re-assignment (inside a loop) is still refused
        g("for period_start in [1]: pass")


def test_could_not_index_error_gets_column_hint():
    def inner(code):
        raise RuntimeError("InterpreterError: Could not index <df> with 'Thành tiền'")
    g = GuardedExecutor(inner)
    g.tables = {"don_hang": pd.DataFrame({"Mã đơn": [1], "Tổng bill": [2]})}
    with pytest.raises(RuntimeError, match="has columns"):
        g("x = tables['don_hang']['Thành tiền'].sum()")
    with pytest.raises(ValueError, match="no table 'item_ref'"):      # refused before running, with the real names
        g("x = tables['item_ref']['price_on']")


def test_pasted_literal_after_computed_answer_is_dropped():
    seen = []
    g = GuardedExecutor(lambda code: seen.append(code) or "ran")
    assert g("df = tables['a']\nrevenue = df['x'].sum()\nanswer = revenue\nanswer = 230034300") == "ran"
    assert "230034300" not in seen[-1] and g.dropped_literals == 1


def test_literal_answer_allowed_once_printed():
    out = SimpleNamespace(logs="Doanh thu: 230,034,300", output=None)
    g = GuardedExecutor(lambda code: out)
    g("df = tables['a']\nprint(df['x'].sum())")
    assert g("df = tables['a']\nn = len(df)\nanswer = 230034300") is out  # value was printed before: not a guess
    with pytest.raises(ValueError):
        g("df = tables['a']\nn = len(df)\nanswer = 999")


def test_date_comparison_and_missing_import_hints():
    for msg, needle in (("TypeError: Invalid comparison between dtype=datetime64[us] and date", "pd.Timestamp"),
                        ("InterpreterError: The variable `unicodedata` is not defined.", "import unicodedata")):
        def inner(code, msg=msg):
            raise RuntimeError(msg)
        with pytest.raises(RuntimeError, match=needle.replace(".", r"\.")):
            GuardedExecutor(inner)("x = 1 + 1")


def test_rolling_window_is_n_days_including_today():
    """Rulebook: "N ngày gần nhất" = N days, the last one today. The model wrote 31 days (dev300 q_6f877dfe79e8)."""
    from datamate.agents.explainer import repair_spec
    spec = {"rewritten_question": "Doanh thu 30 ngày gần nhất", "answer_type": "number", "measure": "money",
            "periods": [{"label": "30 ngày gần nhất", "start": "2026-05-19", "end": "2026-06-18", "role": "main"}]}
    s, errs, _, notes = repair_spec(spec, {}, "2026-06-18", None, question="dthu 30 ngay gan nhat cua cua hang la bao nhieu")
    assert (s.periods[0]["start"], s.periods[0]["end"]) == ("2026-05-20", "2026-06-18") and any("30 ngày" in n for n in notes)
    ok = {**spec, "periods": [{**spec["periods"][0], "start": "2026-05-20"}]}
    s, _, _, notes = repair_spec(ok, {}, "2026-06-18", None, question="doanh thu 30 ngày gần nhất")
    assert s.periods[0]["start"] == "2026-05-20" and not any("30 ngày" in n for n in notes)


def test_month_to_date_compare_uses_same_number_of_days():
    from datamate.agents.explainer import repair_spec
    spec = {"rewritten_question": "x", "answer_type": "percent", "measure": "money", "periods": [
        {"label": "tháng này", "start": "2025-08-01", "end": "2025-08-17", "role": "main"},
        {"label": "tháng trước", "start": "2025-07-01", "end": "2025-07-31", "role": "compare"}]}
    s, _, _, notes = repair_spec(spec, {}, "2025-08-17", None, question="so với cùng kỳ tháng trước, doanh thu tháng này")
    assert s.periods[1]["end"] == "2025-07-17" and notes
    spec2 = {**spec, "periods": [
        {"label": "tháng này", "start": "2025-08-01", "end": "2025-08-17", "role": "main"},
        {"label": "tháng trước", "start": "2025-07-01", "end": "2025-07-31", "role": "compare"}]}
    s, _, _, _ = repair_spec(spec2, {}, "2025-08-17", None, question="so với cả tháng trước thì sao")
    assert s.periods[1]["end"] == "2025-07-31"


def test_dtype_helpers_are_rewritten_and_table_names_fixed_or_refused():
    seen = []
    g = GuardedExecutor(lambda code: seen.append(code) or "ran")
    g.tables = {"Don_Hang": pd.DataFrame({"a": [1]})}
    assert g("import pandas as pd\nok = pd.api.types.is_numeric_dtype(tables['don_hang']['a'])") == "ran"
    assert "pd.api" not in seen[-1] and "is_numeric_dtype(" in seen[-1] and "tables['Don_Hang']" in seen[-1]   # spelling fixed in place
    with pytest.raises(ValueError, match="no table 'khach'"):
        g("x = tables['khach']")


def test_item_attribute_merge_and_valid_as_cancelled_are_bounced_once():
    lines = pd.DataFrame({"order": ["O1"], "item": ["a"], "amount": [1.0], "Thuốc kê đơn": [False]})
    orders = pd.DataFrame({"order": ["O1"], "valid": [True]})
    g = GuardedExecutor(lambda code: "ran")
    g.tables, g.frames = {"don": orders}, {"item_lines": lines, "sales_orders": orders}
    merge = "m = tables['don'].merge(tables['don'], left_on='a', right_on='b')\nx = m['Thuốc kê đơn']"
    with pytest.raises(ValueError, match="item_lines"):
        g(merge)
    assert g(merge) == "ran"                                           # a second attempt runs as written
    g2 = GuardedExecutor(lambda code: "ran")
    g2.tables, g2.frames = {"don": orders}, {"item_lines": lines, "sales_orders": orders}
    assert g2("x = item_lines[item_lines['Thuốc kê đơn'] == False]['amount'].sum()") == "ran"      # item_lines used: fine
    cancelled = "n = sales_orders[~sales_orders['valid']]['order'].nunique()"
    with pytest.raises(ValueError, match="status_std"):
        g2(cancelled)
    assert g2(cancelled) == "ran"
    g3 = GuardedExecutor(lambda code: "ran")
    g3.tables, g3.frames = {"don": orders}, {"sales_orders": orders}
    assert g3("n = sales_orders[sales_orders['status_std'] == 'cancelled']['order'].nunique()\nm = ~sales_orders['valid']") == "ran"
