import pandas as pd

from datamate.data.loader import load_business
from datamate.engines.sales import build_sales

README = """# biz

## Cột trong `chi_tiet.csv`
- **Tiền bán**: thành tiền sau giảm
- **Số CT**: mã đơn

## Cột trong `don_hang.csv`
- **Ngày bán**: ngày bán
- **Số phiếu**: mã đơn
- **Tình trạng đơn**: trạng thái đơn
"""


def make(tmp_path):
    d = tmp_path / "biz"
    d.mkdir()
    (d / "README.md").write_text(README, encoding="utf-8")
    pd.DataFrame({  # O1 ok (2 lines), O2 cancelled, O3 ok, O4 returned via side table, O5 next day; "Cộng ngày" row = summary
        "Số CT": ["O1", "O1", "O2", "O3", "O4", "O5", None],
        "Tiền bán": ["100.000đ", "1,5tr", "999.000đ", "200.000đ", "300.000đ", "50.000đ", "5tr"],
        "Hàng": ["a", "b", "c", "d", "e", "f", "Cộng ngày 01/03"],
    }).to_csv(d / "chi_tiet.csv", index=False)
    pd.DataFrame({
        "Số phiếu": ["O1", "O2", "O3", "O4", "O5"],
        "Ngày bán": ["01/03/2025", "01/03/2025", "01/03/2025", "01/03/2025", "02/03/2025"],
        "Tình trạng đơn": ["Đã giao", "Đã hủy", "xong", "xong", "xong"],
    }).to_csv(d / "don_hang.csv", index=False)
    pd.DataFrame({"Mã đơn hàng": ["O4"], "Số lượng trả": [1.0]}).to_csv(d / "tra_hang.csv", index=False)
    return load_business(d)


def test_sales_view_follows_the_rulebook(tmp_path):
    biz = make(tmp_path)
    v = build_sales(biz)
    assert v is not None and v.source == "readme" and v.table == "chi_tiet" and v.amount_col == "Tiền bán"
    day1 = v.metrics("2025-03-01", "2025-03-01")
    # O1 = 100k + 1.5M, O3 = 200k; O2 cancelled, O4 returned (side table), summary row dropped
    assert day1 == {"revenue": 1_800_000, "orders": 2, "avg_order": 900_000, "lines": 3}
    assert v.metrics("2025-03-01", "2025-03-02")["revenue"] == 1_850_000   # O5 on the next day
    assert v.metrics("2025-04-01", "2025-04-30")["orders"] == 0


def test_voucher_points_and_shipping_rows_are_not_revenue(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    (d / "README.md").write_text("## Cột trong `ban.csv`\n- **TT**: thành tiền của dòng; voucher, dùng điểm ghi số âm\n- **HĐ**: mã đơn\n- **Ngày**: ngày bán\n", encoding="utf-8")
    pd.DataFrame({
        "HĐ": ["H1", "H1", "H1", "H1", "H2"], "Ngày": ["01/03/2025", None, None, None, "01/03/2025"],
        "Tên hàng": ["Sữa", "Voucher", "Dùng 12 điểm", "Phí ship", "Bỉm"], "TT": [500000, -50000, -1200, 30000, 200000],
    }).to_csv(d / "ban.csv", index=False)
    v = build_sales(load_business(d))
    assert v.metrics("2025-03-01", "2025-03-01") == {"revenue": 700000, "orders": 2, "avg_order": 350000, "lines": 2}


def test_summary_label_in_a_late_column_is_dropped(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    cols = {f"c{i}": ["x", "y", "z"] for i in range(14)}
    cols["Họ tên khách"] = ["An", "TỔNG NGÀY 31/05", "Bình"]   # 15th column
    cols["Thành tiền"] = [10, 999, 20]
    pd.DataFrame(cols).to_csv(d / "ban_2025.csv", index=False)
    (df,) = load_business(d).tables.values()
    assert df["Thành tiền"].tolist() == [10, 20]
