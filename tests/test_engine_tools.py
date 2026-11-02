"""Engines as coder tools: correct figures, loud misuse, callable from the guarded sandbox, engines no longer answer."""
import pandas as pd
import pytest

from datamate.engines.functions import build_tools
from datamate.engines.items import ItemView
from datamate.engines.sales import SalesView

SALES = SalesView(pd.DataFrame({"order": ["A", "A", "B", "C"], "date": pd.to_datetime(["2026-05-01", "2026-05-01", "2026-05-02", "2026-05-03"]),
                                "amount": [100.0, 50.0, 200.0, 999.0], "valid": [True, True, True, False]}),
                  "ban", "Thành tiền", "Mã đơn", "Ngày", "readme")
LINES = pd.DataFrame({
    "order": ["A", "B", "B", "C"], "date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-02", "2026-05-03"]),
    "valid": [True, True, True, False], "is_item": [True] * 4, "item": ["Bánh tiramisu", "Bánh tiramisu", "Cà phê sữa (M)", "Bánh tiramisu"],
    "key": ["banh tiramisu", "banh tiramisu", "ca phe sua (m)", "banh tiramisu"], "code": ["T1", "T1", "C1", "T1"],
    "qty": [2.0, 1.0, 3.0, 5.0], "qty_base": [2.0, 1.0, 3.0, 5.0], "unit": ["cái"] * 3 + ["cái"], "base_unit": ["cái"] * 4,
    "amount": [60.0, 30.0, 75.0, 150.0], "unit_price": [30.0, 30.0, 25.0, 30.0]})
ITEMS = ItemView(LINES, "ban", "")


def test_tools_compute_rulebook_figures():
    t = build_tools(SALES, ITEMS)
    assert t["sales_metrics"]("2026-05-01", "2026-05-31") == {"revenue": 350, "orders": 2, "avg_order": 175}  # C cancelled
    assert t["item_qty"]("banh tiramisu", "2026-05-01", "2026-05-31") == 3                                   # C excluded
    assert t["item_revenue"]("Bánh tiramisu", "2026-05-01", "2026-05-31") == 90
    assert t["top_items"](1, "2026-05-01", "2026-05-31")[0] == ("Cà phê sữa (M)", 3)  # tie at 3: lower item code first
    assert t["top_items"](1, "2026-05-01", "2026-05-31", by="amount")[0] == ("Bánh tiramisu", 90)
    assert t["find_items"]("tiramisu") == ["Bánh tiramisu"]
    assert t["item_price_on"]("banh tiramisu", "2026-05-02") == 30.0


def test_unknown_item_fails_loudly_with_close_names():
    with pytest.raises(ValueError, match="closest names") as e:
        build_tools(SALES, ITEMS)["item_qty"]("banh tiramisuu", "2026-05-01", "2026-05-31")
    assert "Bánh tiramisu" in str(e.value)


def test_coder_can_call_tools_and_cannot_overwrite_them():
    from smolagents.local_python_executor import LocalPythonExecutor

    from datamate.sandbox.guard import PROTECTED, GuardedExecutor
    inner = LocalPythonExecutor(additional_authorized_imports=[])
    inner.send_tools({})
    inner.send_variables(build_tools(SALES, ITEMS))
    ex = GuardedExecutor(inner)
    out = ex("n = sales_metrics('2026-05-01', '2026-05-31')['orders']\nprint(n)")
    assert "2" in out.logs
    assert "sales_metrics" in PROTECTED
    with pytest.raises(ValueError):
        ex("sales_metrics = lambda a, b: {'orders': 99}")


def test_cancel_rate_counts_every_status_in_the_denominator():
    from pathlib import Path

    from datamate.engines.functions import rule_gated
    orders = pd.DataFrame({"order": list("ABCDE"),
                           "date": pd.to_datetime(["2026-05-01 09:00", "2026-05-10 00:00", "2026-05-20 00:00", "2026-05-31 20:30", "2026-06-01 00:00"]),
                           "status_std": ["completed", "cancelled", "returned", "shipping", "cancelled"]})
    t = build_tools(None, None, orders=orders)
    assert t["cancel_rate"]("2026-05-01", "2026-05-31") == 25.0        # 1 of 4: returned is not cancelled, 31st 20:30 is in
    assert t["cancel_rate"](pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01 23:59:59")) == 100.0
    assert t["cancel_rate"]("2026-07-01", "2026-07-31") is None         # no order placed: no data, not 0 (§7)
    assert "cancel_rate" not in build_tools(None, None, orders=orders.drop(columns="status_std"))
    # the engine is offered only when the split's rulebook still defines the rate this way
    assert rule_gated("RULEBOOK") == {"cancel_rate"}
    row = "| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**: cả đơn huỷ) |"
    assert rule_gated(row) == set()
    book = Path(__file__).resolve().parents[1] / "train" / "rulebook.md"
    if book.exists():
        assert rule_gated(book.read_text(encoding="utf-8")) == set()



def test_average_order_rounds_half_a_dong_up():
    sales = SalesView(pd.DataFrame({"order": ["A", "B"], "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
                                    "amount": [100.0, 251.0], "valid": [True, True]}), "ban", "Thành tiền", "Mã đơn", "Ngày", "readme")
    assert build_tools(sales, None)["sales_metrics"]("2026-05-01", "2026-05-31")["avg_order"] == 176    # 175.5 -> 176 (round: 176 too)
    sales2 = SalesView(pd.DataFrame({"order": ["A", "B"], "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
                                     "amount": [100.0, 249.0], "valid": [True, True]}), "ban", "Thành tiền", "Mã đơn", "Ngày", "readme")
    assert build_tools(sales2, None)["sales_metrics"]("2026-05-01", "2026-05-31")["avg_order"] == 175   # 174.5 -> 175 (round: 174)
