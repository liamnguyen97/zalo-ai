import pandas as pd

from datamate.data.descriptor import describe_column, describe_relations, describe_table
from datamate.data.status import add_status_std


def test_low_cardinality_lists_all_values_with_counts():
    s = pd.Series(["completed"] * 5 + ["cancelled"] * 2 + ["shipping"] + ["returned"] * 3)
    d = describe_column(s, "Tình trạng")
    assert all(v in d for v in ("completed(5)", "cancelled(2)", "shipping(1)", "returned(3)")) and "ALL values" in d
    ids = pd.Series([f"ORD{i:05d}" for i in range(500)])
    assert "ALL values" not in describe_column(ids, "Mã đơn") and "e.g." in describe_column(ids, "Mã đơn")


def test_status_std_mapping_is_shown_on_the_card():
    df = pd.DataFrame({"Mã đơn": ["A", "B", "C"], "Trạng thái": ["Đã giao", "Đang vận chuyển", "Đổi trả"]})
    card = describe_table("don", add_status_std(df))
    assert "status_std" in card and "completed" in card and "shipping" in card and "returned" in card


def test_relations_find_keys_with_different_names():
    ids = [f"ORD{i:06d}" for i in range(300)]
    orders = pd.DataFrame({"Mã chứng từ": ids, "Tổng bill": range(300)})
    lines = pd.DataFrame({"Ma don": ids * 3, "SL": 1})
    other = pd.DataFrame({"Mã hàng": [f"SP{i:04d}" for i in range(50)], "ten": "x"})
    rel = describe_relations({"don_hang": orders, "chi_tiet_don": lines, "san_pham": other})
    assert "`don_hang`.`Mã chứng từ` ↔ `chi_tiet_don`.`Ma don`" in rel
    assert "san_pham" not in rel  # unrelated table: no false join hint


def test_card_marks_line_level_tables_and_valid_order():
    ids = [f"O{i:04d}" for i in range(100)]
    lines = pd.DataFrame({"Mã đơn": ids * 3, "Trạng thái": ["Đã giao", "Đang giao", "Đã hủy"] * 100, "SL": 1})
    card = describe_table("ct", add_status_std(lines))
    assert "COUNT ORDERS = `Mã đơn`.nunique()" in card and "valid_order" in card
