"""Purchases engine (engines/purchases.py): slips + lines and flat layouts, weighted average price, cancelled slips, tools."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from datamate.data.loader import Business
from datamate.data.structure import detect_structure
from datamate.engines.functions import build_tools
from datamate.engines.purchases import build_purchases
from datamate.engines.reference import build_reference, clean_call
from datamate.text import fold

N_ITEMS = 12


def _items(day="2025-12-31"):
    names = [f"Hàng số {k}" for k in range(N_ITEMS)]
    lines = pd.DataFrame({"code": [f"SP{k:04d}" for k in range(N_ITEMS)], "key": [fold(n) for n in names], "item": names,
                          "unit": "thùng", "base_unit": "lon", "date": pd.Timestamp(day), "is_item": True})
    return SimpleNamespace(lines=lines, table="DanhSachHoaDonChiTiet", display=dict(zip(lines["key"], names)))


def _slips_shop(seed=1, n_slips=80):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_slips):
        for _ in range(int(rng.integers(2, 6))):
            rows.append({"Mã nhập hàng": f"PN{i:05d}", "Mã hàng": f"SP{int(rng.integers(N_ITEMS)):04d}",
                         "ĐVT": str(rng.choice(["thùng", "lốc"], p=[0.8, 0.2])), "Số lượng": float(rng.integers(1, 20)),
                         "Đơn giá": float(rng.integers(10, 300)) * 1000})
    L = pd.DataFrame(rows)
    tot = (L["Số lượng"] * L["Đơn giá"]).groupby(L["Mã nhập hàng"]).sum()
    H = pd.DataFrame({"Mã nhập hàng": tot.index, "Thời gian": pd.date_range("2025-01-01", periods=len(tot), freq="D"),
                      "Mã NCC": [f"NCC{i % 4}" for i in range(len(tot))], "Cần trả NCC": tot.values,
                      "Trạng thái": ["Đã hủy" if i % 10 == 3 else "Đã nhập hàng" for i in range(len(tot))]})
    sales = pd.DataFrame({"Mã hóa đơn": [f"HD{k}" for k in range(300)], "Mã hàng": [f"SP{k % N_ITEMS:04d}" for k in range(300)],
                          "Đơn giá": 5000.0})
    return {"File__DanhSachNhapHang": H, "File__DanhSachChiTietNhapHang": L, "DanhSachHoaDonChiTiet": sales}, H, L


def _view(tables):
    biz = Business("b", "- **Nhập hàng**: **Đơn giá** là giá nhập một đơn vị.", tables)
    return build_purchases(biz, _items(), detect_structure(tables, biz.readme))


def test_slips_layout_weighted_average_excludes_cancelled_and_uses_the_asked_unit():
    tables, H, L = _slips_shop()
    pv = _view(tables)
    assert pv.layout == "slips" and pv.cols["total"] == "Cần trả NCC" and pv.cols["unit"] == "ĐVT"
    t = build_tools(None, _items(), purchases=pv)
    j = L.merge(H, on="Mã nhập hàng")
    j = j[(j["Trạng thái"] != "Đã hủy") & (j["Thời gian"] >= "2025-01-10") & (j["Thời gian"] <= "2025-02-20")
          & (j["Mã hàng"] == "SP0003") & (j["ĐVT"] == "thùng")]
    expect = round((j["Số lượng"] * j["Đơn giá"]).sum() / j["Số lượng"].sum())
    assert t["purchase_avg_price"](item="Hàng số 3", start="2025-01-10", end="2025-02-20", unit="thùng") == expect
    # the slip total divided by the quantity (the failure on q_00a6e23cbacf) is a different, much larger number
    assert j["Cần trả NCC"].sum() / j["Số lượng"].sum() > 2 * expect


def test_total_counts_valid_slips_in_the_period_and_the_end_day_is_included():
    tables, H, L = _slips_shop()
    t = build_tools(None, _items(), purchases=_view(tables))
    j = L.merge(H, on="Mã nhập hàng")
    j = j[(j["Trạng thái"] != "Đã hủy") & (j["Thời gian"] >= "2025-01-05") & (j["Thời gian"] <= "2025-01-31")]
    assert t["purchase_total"]("2025-01-05", "2025-01-31") == round((j["Số lượng"] * j["Đơn giá"]).sum())
    top = t["top_supplier"](start="2025-01-01", end="2025-03-31", k=2)
    assert len(top) == 2 and top[0][1] >= top[1][1] and top[0][0].startswith("NCC")


def test_loud_misuse_several_units_unknown_item():
    t = build_tools(None, _items(), purchases=_view(_slips_shop()[0]))
    with pytest.raises(ValueError, match="several units"):
        t["purchase_avg_price"](item="Hàng số 3", start="2025-01-01", end="2025-03-31")
    with pytest.raises(ValueError, match="closest names"):
        t["purchase_avg_price"](item="Hàng số 33", start="2025-01-01", end="2025-03-31", unit="thùng")
    assert t["purchase_avg_price"](item="Hàng số 3", start="2030-01-01", end="2030-01-31", unit="thùng") is None


def test_flat_layout_found_from_qty_price_amount_and_the_sold_units():
    rng = np.random.default_rng(3)
    n = 400
    flat = pd.DataFrame({"Số phiếu": [f"PN{i // 4:04d}" for i in range(n)], "Ngày nhập": pd.date_range("2025-01-01", periods=n // 4).repeat(4),
                         "Nhà cung cấp": rng.choice(["Cty A", "Cty B", "Cty C"], n // 4).repeat(4),
                         "Tên mặt hàng": [f"Hàng số {int(k)}" for k in rng.integers(0, N_ITEMS, n)],
                         "ĐVT": "thùng", "SL nhập": rng.integers(1, 20, n).astype(float), "Giá nhập": rng.integers(10, 300, n) * 1000.0})
    flat["Thành tiền"] = flat["SL nhập"] * flat["Giá nhập"]
    flat["Hạn dùng"] = pd.Timestamp("2029-01-01")                               # an expiry date is not the slip date
    biz = Business("b", "- **Nhập hàng**: **Giá nhập** là giá nhập một đơn vị.", {"SoNhapHang": flat})
    pv = build_purchases(biz, _items(), detect_structure(biz.tables, biz.readme))
    assert pv.layout == "flat" and pv.cols["date"] == "Ngày nhập" and pv.cols["unit"] == "ĐVT" and pv.cols["qty"] == "SL nhập"
    t = build_tools(None, _items(), purchases=pv)
    s = flat[(flat["Tên mặt hàng"] == "Hàng số 5") & (flat["Ngày nhập"] <= "2025-02-15")]
    assert t["purchase_avg_price"](item="Hàng số 5", start="2025-01-01", end="2025-02-15", unit="thùng") == round(s["Thành tiền"].sum() / s["SL nhập"].sum())
    assert t["top_supplier"](start="2025-01-01", end="2025-12-31", k=1)[0][0] in {"Cty A", "Cty B", "Cty C"}


def test_a_lots_table_without_prices_gives_no_engine():
    # biz_121 / biz_122: purchase lines hold quantities and lots only: no price, no amount -> nothing to compute, no guess
    tables, H, L = _slips_shop()
    L2 = L.drop(columns=["Đơn giá"])
    assert _view({**tables, "File__DanhSachChiTietNhapHang": L2}) is None


def test_the_engine_calls_are_valid_references():
    tables, H, L = _slips_shop()
    t = build_tools(None, _items(), purchases=_view(tables))
    call = {"function": "purchase_avg_price", "args": {"item": "Hàng số 3", "start": "2025-01-01", "end": "2025-03-31", "unit": "thùng"}}
    assert clean_call(call) == call and build_reference(call, t).value == t["purchase_avg_price"](**call["args"])
    assert clean_call({"function": "purchase_total", "args": {"start": "2025-01-01", "end": "bad"}}) is None
    assert "purchase_total" not in build_tools(None, _items())                     # no purchases: no function


REAL = Path(__file__).resolve().parents[1] / "train" / "data"


@pytest.mark.skipif(not (REAL / "biz_109").exists() or os.getenv("DATAMATE_DATA_TESTS") != "1",
                    reason="reads real shops (~15 s): set DATAMATE_DATA_TESTS=1 with train/ unpacked")
def test_real_shops_match_the_gold_answers():
    from datamate.context import BusinessContext
    for biz, args, gold in [("biz_109", dict(item="Mì Omachi sườn hầm", start="2025-01-01", end="2025-01-31", unit="thùng"), 200556),
                            ("biz_109", dict(item="Nước suối Lavie 1.5L", start="2025-03-01", end="2025-03-31", unit="thùng"), 97765),
                            ("biz_105", dict(item="Bánh croissant", start="2025-12-01", end="2025-12-31", unit="hộp"), 87081),    # flat
                            ("biz_123", dict(item="Panadol Extra hộp 15 vỉ x 12 viên", start="2026-04-01", end="2026-04-30", unit="hộp"), 192000)]:
        ctx = BusinessContext.build(REAL / biz)
        t = build_tools(ctx.sales, ctx.items, ctx.debt, ctx.orders, ctx.purchases)
        assert t["purchase_avg_price"](**args) == gold, biz
