"""Shop books with unusual shapes (error-analysis P2.1, P2.2, P3.1a): running order numbers, one row per slip, abbreviation sheets,
nicknames of staff. Synthetic data; the public-shop checks run only with DATAMATE_DATA_TESTS=1 and DATAMATE_PUBLIC_DATA=<public_test/data>."""
import os
from pathlib import Path

import pandas as pd
import pytest

from datamate.data.loader import Business, derive_order_ids, load_business
from datamate.engines.orders import build_orders
from datamate.engines.sales import build_sales


def _book(blank_continuation: bool = True, n_orders: int = 30) -> pd.DataFrame:
    """A shop book: STT written once per order (later lines blank) or on every line; 1-3 lines per order."""
    rows = []
    for i in range(1, n_orders + 1):
        day = pd.Timestamp("2025-06-01") + pd.Timedelta(days=i // 4)
        for j in range(1 + i % 3):
            first = j == 0 or not blank_continuation
            rows.append({"Ngày": day, "STT": float(i) if first else None,
                         "Khách": (f"Khách {i}" if i % 2 else "lẻ") if first else None,
                         "Hàng": f"Hàng {j}", "SL": 1.0, "Thành tiền": 1000.0 * (j + 1),
                         "Khách trả": 5000.0 if first else None,
                         "Ghi chú": ("khách trả lại" if i == 5 else None) if first else None})
    return pd.DataFrame(rows)


def test_running_number_with_blank_lines_becomes_the_order_id():
    df = _book(blank_continuation=True)
    out, col = derive_order_ids(df)
    assert col == "Mã đơn (STT)"
    assert out[col].nunique() == 30 == df["Khách trả"].notna().sum()      # one order per first line
    assert out[col].notna().all() and out[col].iloc[0] == "1"              # the number itself: other books refer to it


def test_running_number_on_every_line():
    out, col = derive_order_ids(_book(blank_continuation=False))
    assert col is not None and out[col].nunique() == 30


def test_book_restarting_each_day_is_keyed_by_day():
    df = _book(blank_continuation=False, n_orders=40)
    df["STT"] = df.groupby("Ngày")["STT"].rank(method="dense")            # 1, 2, ... again every day
    out, col = derive_order_ids(df)
    assert col is not None and out[col].nunique() == 40 and out[col].str.contains("/").all()


def test_not_an_order_number():
    df = _book(blank_continuation=False)
    plain = df.assign(STT=range(1, len(df) + 1))                            # a row counter: every row its own number
    assert derive_order_ids(plain)[1] is None
    day = df.assign(STT=df["Ngày"].dt.day.astype(float))                    # a day-of-month column moves with the date
    assert derive_order_ids(day)[1] is None
    lines = df.assign(**{"Mã phiếu": df["STT"].ffill().map(lambda x: f"PN{int(x):04d}")},
                      STT=df.groupby(df["STT"].ffill()).cumcount() + 1.0)   # line numbers inside a slip that has its own id
    assert derive_order_ids(lines)[1] is None
    assert derive_order_ids(df.assign(**{"Mã đơn": df["STT"].ffill().astype(str)}))[1] is None   # a real order id wins


def test_loader_fills_the_order_and_its_status(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    df = _book(blank_continuation=True)
    with pd.ExcelWriter(d / "SoBanHang.xlsx") as w:
        df.assign(Ngày=df["Ngày"].dt.strftime("%d/%m/%Y")).to_excel(w, sheet_name="Sổ bán hàng", index=False)
    (t,) = load_business(d).tables.values()
    five = t[t["Mã đơn (STT)"] == "5"]
    assert len(five) == 3 and five["Khách"].eq("Khách 5").all()            # the order's cells copied to its later lines
    assert five["status_std"].eq("returned").all()                          # a returned order: every line of it
    biz = load_business(d)
    orders = build_orders(biz, build_sales(biz))
    assert len(orders) == 30 and int(orders["valid"].sum()) == 29


def test_footnote_rows_are_not_records(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    days = [f"{i:02d}/06/2025" for i in range(1, 21)]
    raw = pd.DataFrame({"Ngày": days + ["Ghi chú: số liệu ghi tay"], "Xe": [f"wave 59B1-{i:03d}.11" for i in range(20)] + [None],
                        "Tiền": ["120.000đ"] * 20 + [None]})
    raw.to_csv(d / "so.csv", index=False)
    (t,) = load_business(d).tables.values()
    assert len(t) == 20          # the footnote's only text was in the date column: nothing is left of it


def test_one_row_per_slip_book():
    t = pd.DataFrame({"Ngày": pd.date_range("2025-07-01", periods=40, freq="D"), "Xe": [f"wave 59B1-{i:03d}.11" for i in range(40)],
                      "Làm gì": ["thay nhớt"] * 40, "Thợ": ["a Đức"] * 40, "Tiền": [100000.0] * 39 + [0.0]})
    biz = Business("x", "Sổ sửa xe: mỗi dòng một lượt xe vào sửa (một phiếu), không ghi mã phiếu, không ghi tên khách.", {"SoSuaXe": t})
    sales = build_sales(biz)
    assert sales.row_docs and sales.order_col is None
    o = build_orders(biz, sales)
    assert len(o) == 40 and o["walk_in"].all()                              # the README says it records no customer
    assert sales.metrics("2025-07-01", "2025-08-09")["orders"] == 40


def test_line_ledger_without_order_ids_has_no_orders():
    t = pd.DataFrame({"ngay_ban": pd.date_range("2025-07-01", periods=40, freq="D"), "Nội dung": ["Coca"] * 40,
                      "Số lượng": [1.0] * 40, "Tổng tiền": [10000.0] * 40})
    biz = Business("x", "Sổ bán hàng: mỗi dòng một mặt hàng khách mua, không ghi mã đơn, không ghi tên khách.", {"SoBanHang": t})
    sales = build_sales(biz)
    assert not sales.row_docs and build_orders(biz, sales) is None


PUBLIC = Path(os.getenv("DATAMATE_PUBLIC_DATA", "/nonexistent"))


@pytest.mark.skipif(not (PUBLIC / "dev_011").exists() or os.getenv("DATAMATE_DATA_TESTS") != "1",
                    reason="reads public shops: set DATAMATE_DATA_TESTS=1 and DATAMATE_PUBLIC_DATA=<public_test/data>")
def test_public_phone_and_repair_shops_get_orders():
    from datamate.context import BusinessContext
    c11 = BusinessContext.build(PUBLIC / "dev_011")
    t = c11.tables[c11.sales.table]
    assert c11.onboarding["trusted"] and len(c11.orders) == 3819 == int(t["Khách trả"].notna().sum())
    c14 = BusinessContext.build(PUBLIC / "dev_014")
    assert c14.onboarding["trusted"] and len(c14.orders) == 6334
    assert "Mã đơn (STT)" not in c14.tables["SoNo__Trả_góp"].columns    # its STT is the sales order of each instalment
    c12 = BusinessContext.build(PUBLIC / "dev_012")
    assert c12.onboarding["trusted"] and len(c12.orders) == 6060 and c12.orders["walk_in"].all()


def _phone_book(tmp_path, with_sheet: bool = True):
    d = tmp_path / "biz"; d.mkdir()
    sheet = pd.DataFrame({"Ghi tắt": ["Sạc 25W SS", "Cáp C-L", "IP16 128 đen"],
                          "Tên hàng": ["Củ sạc nhanh 25W Samsung", "Cáp Type-C sang Lightning 1m", "iPhone 16 128GB Đen"],
                          "Loại": ["phụ kiện", "phụ kiện", "máy mới"]})
    names = ["Sạc 25W SS", "cáp c-l", "Củ sạc nhanh 25W Samsung", "IP16  128 den"] * 10
    book = pd.DataFrame({"Ngày": [f"{1 + i % 28:02d}/06/2025" for i in range(40)], "STT": [float(i + 1) for i in range(40)],
                         "Hàng": names, "SL": [1.0] * 40, "Thành tiền": [100000.0] * 40})
    with pd.ExcelWriter(d / "SoBanHang.xlsx") as w:
        if with_sheet:
            sheet.to_excel(w, sheet_name="Ghi tắt", index=False)
        book.to_excel(w, sheet_name="Sổ bán hàng", index=False)
    return d


def test_abbreviation_sheet_writes_items_in_full(tmp_path):
    from datamate.engines.functions import build_tools
    from datamate.engines.items import build_items
    biz = load_business(_phone_book(tmp_path))
    (e,) = biz.expansions
    t = biz.tables[e.table]
    assert e.column == "Hàng" and t["Hàng"].iloc[0] == "Sạc 25W SS"                # the book's column is left as written
    assert t["Hàng (tên đầy đủ)"].tolist()[:4] == ["Củ sạc nhanh 25W Samsung", "Cáp Type-C sang Lightning 1m",
                                                   "Củ sạc nhanh 25W Samsung", "iPhone 16 128GB Đen"]
    assert t["Loại"].iloc[3] == "máy mới"
    items = build_items(biz)
    tools = build_tools(None, items)
    assert tools["item_qty"]("Củ sạc nhanh 25W Samsung", "2025-06-01", "2025-06-30") == 20
    assert tools["item_qty"]("Sạc 25W SS", "2025-06-01", "2025-06-30") == 20          # the short form resolves to the same item
    assert tools["find_items"]("25w ss") == ["Củ sạc nhanh 25W Samsung"]


def test_a_product_catalogue_is_not_an_abbreviation_sheet():
    from datamate.data.aliases import find_expansions
    cat = pd.DataFrame({"Mã hàng": ["SP1", "SP2", "SP3"], "Tên hàng": ["Coca lon", "Pepsi lon", "Bia Tiger"],
                        "Nhóm hàng": ["Nước ngọt", "Nước ngọt", "Bia"]})
    lines = pd.DataFrame({"Mã hàng": ["SP1", "SP2", "SP3"] * 10, "Tên hàng": ["Coca lon", "Pepsi lon", "Bia Tiger"] * 10})
    assert find_expansions({"DanhMuc": cat, "ChiTiet": lines}) == []


def test_titles_before_a_given_name():
    from datamate.text import given_name
    assert [given_name(x) for x in ("A.Đức", "a Đức", "Đức", "chị Lan", "c.Hoa")] == ["duc", "duc", "duc", "lan", "hoa"]
    assert [given_name(x) for x in ("Anh", "An", "Lương Chi Đức")] == ["anh", "an", "luong chi duc"]   # nothing after it: no title


def test_staff_nicknames_get_the_roster_full_name():
    from datamate.data.aliases import expand_nicknames
    book = pd.DataFrame({"Thợ": ["A.Đức", "a Đức", "Đức", "a Phúc", "An"] * 6, "Tiền": [1.0] * 30})
    roster = pd.DataFrame({"Họ tên": ["Lương Chi Đức", "Đoàn Bảo Phúc", "Đàm Lan An"], "Hoa hồng (%)": [25.0, 28.0, 33.0]})
    out, (n,) = expand_nicknames({"SoSuaXe": book, "DanhSachTho": roster})
    assert out["SoSuaXe"]["Thợ (họ tên)"].tolist()[:5] == ["Lương Chi Đức"] * 3 + ["Đoàn Bảo Phúc", "Đàm Lan An"]
    two_duc = roster.assign(**{"Họ tên": ["Lương Chi Đức", "Đoàn Bảo Phúc", "Trần Minh Đức"]})   # a given name twice: no match
    assert expand_nicknames({"SoSuaXe": book, "DanhSachTho": two_duc})[1] == []
