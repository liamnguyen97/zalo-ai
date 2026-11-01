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
