import pandas as pd

from datamate.data.loader import load_business
from datamate.engines.items import build_items
from datamate.text import fold

README = """# biz

## Cột trong `ban.csv`
- **Mã đơn hàng**: mã đơn
- **Ngày**: ngày bán
- **Tên hàng hóa**: tên hàng
- **Đơn vị tính**: đơn vị tính
- **Số lượng**: số lượng
- **Đơn giá**: đơn giá
- **Thành tiền**: thành tiền sau giảm
- **Trạng thái**: trạng thái đơn

## Quy ước
- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng:
  - Bia Tiger lon: 1 thùng = 24 lon
"""


def make(tmp_path):
    d = tmp_path / "biz"
    d.mkdir()
    (d / "README.md").write_text(README, encoding="utf-8")
    rows = [  # order, date, item, unit, qty, price, amount, status
        ("O1", "01/03/2025", "Bia Tiger lon", "lon", 2, 20000, 40000, "xong"),
        ("O1", "01/03/2025", "Cà phê sữa (M)", "ly", 1, 30000, 30000, "xong"),
        ("O2", "02/03/2025", "Bia Tiger lon", "thùng", 1, 480000, 480000, "xong"),      # 1 thùng = 24 lon
        ("O3", "02/03/2025", "BIA  TIGER LON", "lon", 3, 20000, 60000, "xong"),          # dirty spelling of the same item
        ("O4", "02/03/2025", "Bia Tiger lon", "lon", 99, 20000, 1980000, "huỷ"),         # cancelled: ignored
        ("O5", "02/03/2025", "Cà phê sữa (L)", "ly", 4, 35000, 140000, "xong"),
        ("O5", "02/03/2025", "Phí phục vụ", "", 1, 0, 25000, "xong"),                     # not an item
        ("O6", "09/03/2025", "Bia Tiger lon", "lon", 6, 20000, 120000, "xong"),
    ]
    pd.DataFrame(rows, columns=["Mã đơn hàng", "Ngày", "Tên hàng hóa", "Đơn vị tính", "Số lượng", "Đơn giá", "Thành tiền", "Trạng thái"]).to_csv(d / "ban.csv", index=False)
    return load_business(d)


def test_base_units_ranking_size_and_price(tmp_path):
    v = build_items(make(tmp_path))
    assert v is not None and "conversion" in v.note
    # 2 + 24 (one thùng) + 3 (dirty spelling merged by folded name); the cancelled order and the 'Phí phục vụ' line are ignored
    assert v.qty("Bia Tiger lon", "2025-03-01", "2025-03-02") == 29
    assert v.base_unit("Bia Tiger lon") == "lon"
    assert v.top(2, "2025-03-01", "2025-03-02") == [("Bia Tiger lon", 29.0), ("Cà phê sữa (L)", 4.0)]
    assert v.size_qty("M", "2025-03-01", "2025-03-02") == 1 and v.size_qty("l", "2025-03-01", "2025-03-02") == 4
    assert v.revenue("Cà phê sữa (L)", "2025-03-01", "2025-03-02") == 140000
    assert v.price_on("Bia Tiger lon", "2025-03-02") == 20000        # per BASE unit: 480000 per thùng / 24 = 20000


def test_price_change_window(tmp_path):
    v = build_items(make(tmp_path))
    ch = v.change("Bia Tiger lon", "2025-03-09", 7)   # 7 days from 9 Mar: 6 ; 7 days before (2-8 Mar): 24+3 = 27 -> -77.78%
    assert ch == {"before": 27.0, "after": 6.0, "pct": -77.78}

    assert fold("Bia  Tiger LON") == fold("bia tiger lon")


def test_attributes_join_through_folded_name_when_lines_have_no_code():
    """biz_121: sales lines carry only a name, spelled differently from the product list; the drug flag still joins."""
    import pandas as pd
    from datamate.engines.items import with_item_attrs
    from datamate.text import fold
    lines = pd.DataFrame({"item": ["Panadol 500mg HỘP", "panadol  500mg hop", "Gaviscon"], "amount": [1, 2, 3]})
    lines["key"] = lines["item"].map(fold)
    lines["code"] = None
    tables = {"sp": pd.DataFrame({"Mã SKU": ["A", "B"], "Tên sản phẩm": ["Panadol 500mg hộp", "Gaviscon"]}),
              "thuoc": pd.DataFrame({"Mã SKU": ["A", "B"], "Thuốc kê đơn": [True, False]})}
    out = with_item_attrs(lines, tables)
    assert out["Thuốc kê đơn"].tolist() == [True, True, False]
