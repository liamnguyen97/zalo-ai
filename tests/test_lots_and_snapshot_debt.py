"""Business engines from the error analysis P3.1 (case-specific, labelled): lots and expiry (§6), debt written on the order (§4)."""
import pandas as pd
import pytest

from datamate.data.loader import Business
from datamate.engines.debt import build_snapshot_debt, owed_from_note
from datamate.engines.functions import build_tools
from datamate.engines.items import ItemView
from datamate.engines.lots import build_lots


def _items():
    lines = pd.DataFrame({"order": ["O1", "O2"], "date": pd.to_datetime(["2026-05-01", "2026-05-02"]), "valid": [True, True],
                          "is_item": [True, True], "item": ["Panadol", "Oresol"], "key": ["panadol", "oresol"], "code": ["SP1", "SP2"],
                          "qty": [1.0, 1.0], "qty_base": [1.0, 1.0], "unit": ["viên"] * 2, "base_unit": ["viên"] * 2,
                          "amount": [1.0, 1.0], "unit_price": [1.0, 1.0]})
    return ItemView(lines, "ct", "")


def test_lots_remaining_and_expiry():
    lots = pd.DataFrame({"MaLo": ["L1", "L2", "L3"], "MaHangHoa": ["SP1", "SP1", "SP2"],
                         "HanSuDung": pd.to_datetime(["2026-06-10", "2026-09-01", "2026-06-20"]),
                         "SLNhapQuyDoi": [10.0, 5.0, 4.0], "DonGiaNhap": [100.0, 120.0, 50.0]})
    by_lot = pd.DataFrame({"MaHoaDon": ["O1", "O1", "O2", "O3"], "MaLo": ["L1", "L3", "L3", "L1"], "SLQuyDoi": [3.0, 4.0, 1.0, 7.0],
                           "SLTraQuyDoi": [0.0, 1.0, 0.0, 0.0]})
    orders = pd.DataFrame({"order": ["O1", "O2", "O3"], "date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]),
                           "status_std": ["completed", "completed", "cancelled"]})
    biz = Business("x", "", {"lo_hang": lots, "chi_tiet_don_lo": by_lot})
    lv = build_lots(biz, _items(), orders)
    t = build_tools(None, _items(), None, orders, None, lv)
    # L1: 10 - 3 (O3 cancelled takes nothing) = 7 left; L3: 4 - 4 + 1 - 1 = 0 left; L2 expires after the window
    assert t["expiring_lots"]("2026-06-01", 30) == [("Panadol", "L1", "2026-06-10", 7, 700)]
    assert t["expiring_items"]("2026-06-01", 30) == ["Panadol"] and t["expiring_value"]("2026-06-01", 30) == 700
    assert t["nearest_expiry"]("Panadol", "2026-06-01") == "2026-06-10"
    assert t["nearest_expiry"]("Panadol", "2026-06-11") == "2026-09-01"       # L1 has expired by then
    assert t["nearest_expiry"]("Oresol", "2026-06-01") is None                # its only lot is empty


@pytest.mark.parametrize("note, total, owed", [
    ("ghi nợ 1.038.500đ, đã trả đủ", 1038500, 0), ("nợ 1238k, khách ck đủ rồi", 1238000, 0), ("ghi sổ 443 nghìn", 443000, 443000),
    ("ck 913 nghìn, thiếu lại 456.500đ", 1369500, 456500), ("khách đưa trước 500k, còn thiếu 1tr2", 1700000, 1200000),
    ("đưa trước 300k, còn lại ghi sổ", 1000000, 700000), ("trả 200k, nợ 300k", 500000, 300000), ("nợ cả đơn", 777000, 777000),
    ("khách trả 1tr645", 1645000, None), (None, 1, None)])
def test_debt_read_from_the_order_note(note, total, owed):
    assert owed_from_note(note, total) == owed


def test_debt_recorded_on_the_order_is_known_at_the_end_of_the_data_only():
    readme = "## Cột trong `hoa_don.csv`\n- **Mã hóa đơn**: mã đơn\n- **Ngày**: ngày bán\n- **Khách**: tên khách\n- **Nợ**: số khách còn nợ của đơn\n"
    df = pd.DataFrame({"Mã hóa đơn": [f"HD{i}" for i in range(30)], "Ngày": pd.date_range("2026-06-01", periods=30, freq="D"),
                       "Khách": [f"Nguyễn Văn T{chr(97 + i)}" for i in range(26)] + ["Lê Minh A", "Lê Minh B", "Lê Minh C", "Lê Minh D"], "Nợ": [0.0] * 27 + [100.0, 200.0, 300.0]})
    view = build_snapshot_debt(Business("x", readme, {"hoa_don": df}))
    t = build_tools(None, None, view)
    assert t["debt_total"]("2026-06-30") == 600 and t["debt_customers"]("2026-06-30") == 3
    with pytest.raises(ValueError, match="only as of the end of its data"):
        t["debt_total"]("2026-06-15")
    assert "debt_repaid" not in t and "debt_new" not in t          # no dated payment: no flow functions offered
    with pytest.raises(ValueError, match="no dated payments"):
        view.repaid("2026-06-01", "2026-06-30")
