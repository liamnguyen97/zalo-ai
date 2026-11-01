"""Data / date lookup tools, README parsing, text helpers and the sandbox inputs (no model calls)."""
from types import SimpleNamespace

import pandas as pd

from datamate.agents.lookups import DataTools
from datamate.data.readme import parse_conversions, roles_by_section
from datamate.sandbox.variables import Runner
from datamate.text import fold, fold_text, nodia

TABLES = {
    "ban": pd.DataFrame({
        "Ngày": pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-11", "2026-05-12"]),
        "Mã đơn": ["A1", "A2", "A3", "A4"],
        "Tên hàng": ["Dầu gió xanh Con Ó chai 12ml", "Bánh tiramisu", "Bánh tiramisu", "Cà phê sữa (M)"],
        "Khách": ["Chị Lan", "Khách lẻ", "Anh Tú", "Chị Lan"],
    })
}
CAL = pd.DataFrame({"date": ["2026-02-16", "2026-05-20"], "weekday": [0, 2], "lunar_day": [29, 4], "lunar_month": [12, 4],
                    "lunar_year": [2025, 2026], "lunar_leap": [0, 0], "holiday": ["", ""]})


def test_find_values_maps_typed_text_to_exact_data_values():
    out = DataTools(TABLES).find_values("dau gio xanh con o")
    assert out.splitlines()[0].endswith("value='Dầu gió xanh Con Ó chai 12ml'")
    assert "column='Tên hàng'" in out.splitlines()[0]
    assert "no value" in DataTools(TABLES).find_values("xi mang ha tien")


def test_date_tools():
    t = DataTools(TABLES, CAL)
    info = t.date_info("2026-05-20")
    assert "Wednesday" in info and "2026-05-18 .. 2026-05-24" in info and "Q2: 2026-04-01 .. 2026-06-30" in info
    assert "lunar: day 4 month 4" in info
    assert t.shift_date("2026-03-31", months=-1) == "2026-02-28"
    assert t.lunar_to_solar(2025, 12, 29) == "2026-02-16"


def test_sandbox_edits_do_not_leak_into_shared_tables():
    shared = {"ban": TABLES["ban"].copy()}
    st = SimpleNamespace(period=None, tables=shared, as_of="2026-05-20", ctx=SimpleNamespace(items=None, orders=None, tables=shared))
    v = Runner(pd.DataFrame()).variables(st)
    v["tables"]["ban"]["Khách"] = "x"
    assert shared["ban"]["Khách"].iloc[0] == "Chị Lan"
    assert v["period_start"] is None and v["item_lines"] is None and v["sales_orders"] is None


def test_readme_roles_by_section():
    r = roles_by_section("# x\n\n## Cột trong `chi_tiet_don.csv`\n- **Tiền bán**: thành tiền sau giảm\n- **Số CT**: mã đơn\n"
                         "\n## Cột trong `don_hang.csv`\n- **Ngày bán**: ngày bán\n- **Tình trạng đơn**: trạng thái đơn\n")
    assert r[("chi_tiet_don", None)] == {"line_amount": "Tiền bán", "order_id": "Số CT"}
    assert r[("don_hang", None)] == {"order_date": "Ngày bán", "order_status": "Tình trạng đơn"}


def test_readme_item_roles_and_conversion_lines():
    r = roles_by_section("## Cột trong `ban.csv`\n- **Tên hàng hóa**: tên hàng\n- **Số lượng quy đổi**: số lượng đã đổi ra đơn vị cơ bản\n"
                         "- **Đơn vị tính**: đơn vị tính\n- **Mã hàng hóa**: mã hàng\n")
    assert r[("ban", None)] == {"item_name": "Tên hàng hóa", "qty_base": "Số lượng quy đổi", "unit": "Đơn vị tính", "item_code": "Mã hàng hóa"}
    conv = parse_conversions("## Quy ước\n- **Quy đổi đơn vị**:\n  - Bia Tiger lon: 1 thùng = 24 lon\n  - Cát xây tô: 1 xe 4m3 = 4 m3\n")
    assert ("Bia Tiger lon", "thùng", 24.0, "lon") in conv and ("Cát xây tô", "xe 4m3", 4.0, "m3") in conv


def test_two_fold_forms_differ_only_in_punctuation():
    assert nodia("Đặng Hà") == "Dang Ha"
    assert fold("  Ống PVC  D50 (4m) ") == "ong pvc d50 (4m)"          # item keys: punctuation kept
    assert fold_text("Ống PVC D50 (4m)!") == "ong pvc d50 4m"           # typed-text matching: punctuation is noise
