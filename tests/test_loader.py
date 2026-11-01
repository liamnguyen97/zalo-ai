import pandas as pd

from datamate.data.loader import load_business


def test_loader_header_detection_and_numbers(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    raw = pd.DataFrame([["SỔ BÁN HÀNG", None, None], [None, None, None], ["Ngày", "Tên", "Thành tiền"],
                        ["01/02/2025", "Cát", "65,000.0000"], ["02/02/2025", "Đá", "1,250,000.0000"]])
    for sheet in ("T02", "T03"):
        pass
    with pd.ExcelWriter(d / "ban.xlsx") as w:
        raw.to_excel(w, sheet_name="T02", header=False, index=False)
        raw.to_excel(w, sheet_name="T03", header=False, index=False)
    biz = load_business(d)
    (df,) = biz.tables.values()
    assert len(df) == 4 and "source_sheet" in df.columns
    assert df["Thành tiền"].tolist() == [65000, 1250000, 65000, 1250000]
    assert df["Ngày"].iloc[0] == pd.Timestamp("2025-02-01")


def test_loader_ids_and_day_counts_not_coerced(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({"Mã barcode": [8939792877748, 12], "Số ngày được nợ": [30, 60], "Hạn": ["03/03/2025", "02/04/2025"]}).to_csv(d / "a.csv", index=False)
    (df,) = load_business(d).tables.values()
    assert df["Mã barcode"].tolist() == ["8939792877748", "12"]
    assert df["Số ngày được nợ"].tolist() == [30, 60]
    assert pd.api.types.is_datetime64_any_dtype(df["Hạn"])


def test_messy_dates_and_money(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({
        "Ngày lập": ["1/1", "01/01/2025", "1 thg 1 2025", "45658", "2025-01-01"],
        "Tiền": ["1tr440", "300.000đ", "55.000đ", "2tr", "55k"],
    }).to_csv(d / "ban_2025.csv", index=False)  # year hint comes from the file/sheet name
    (df,) = load_business(d).tables.values()
    assert df["Ngày lập"].nunique() == 1 and df["Ngày lập"].iloc[0] == pd.Timestamp("2025-01-01")
    assert df["Tiền"].tolist() == [1440000, 300000, 55000, 2000000, 55000]


def test_year_from_sheet_ffill_and_merge(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    def month(m, rows):
        return pd.DataFrame([["Ma don", "Thời gian", "Khách", "Amount"]] + rows)
    with pd.ExcelWriter(d / "SoBanHang.xlsx") as w:
        month(1, [["A1", "1/1", "Lan", 10], ["A1", None, None, 20], ["A2", "2/1", "Tú", 5]]).to_excel(w, sheet_name="T01-2025", header=False, index=False)
        month(2, [["B1", "3/2", "Hà", 7], ["B1", None, None, 8], ["B2", 45700, "Minh", 9]]).to_excel(w, sheet_name="T02-2025", header=False, index=False)
    biz = load_business(d)
    assert list(biz.tables) == ["SoBanHang"]  # two month sheets merged
    df = biz.tables["SoBanHang"]
    assert set(df["source_sheet"]) == {"SoBanHang__T01_2025", "SoBanHang__T02_2025"}
    assert pd.api.types.is_datetime64_any_dtype(df["Thời gian"])
    assert df["Thời gian"].tolist()[:2] == [pd.Timestamp("2025-01-01")] * 2  # d/m + sheet year, ffilled to 2nd row
    assert df["Khách"].tolist()[:2] == ["Lan", "Lan"]
    assert df["Thời gian"].iloc[5] == pd.Timestamp("2025-02-12")  # Excel serial 45700


def test_split_files_merge_by_base_name(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    for n in ("BanHang_Thang01_2025", "BanHang_Thang02_2025"):
        pd.DataFrame({"Ngày": ["01/01/2025"], "TT": [1]}).to_excel(d / f"{n}.xlsx", index=False)
    pd.DataFrame({"Tên": ["x"]}).to_excel(d / "DanhMuc.xlsx", index=False)
    assert sorted(load_business(d).tables) == ["BanHang", "DanhMuc"]


def test_status_std_labels_and_order_level_return(tmp_path):
    from datamate.data.status import classify
    assert [classify(v) for v in ("Đã giao", "đã lấy", "ok", "Hoàn thành", "completed")] == ["completed"] * 5
    assert [classify(v) for v in ("Đang vận chuyển", "Đang xử lý", "Chờ giao", "shipping", "Đang giao dịch")] == ["shipping"] * 5
    assert [classify(v) for v in ("khách hủy", "Đã hủy đơn", "cancel", "Không giao được")] == ["cancelled"] * 4
    assert [classify(v) for v in ("Đổi trả", "Đã hoàn hàng", "hàng trả về", "returned", "Khách trả")] == ["returned"] * 5
    assert classify("màn ám") is None and classify("sửa tại tiệm") is None  # device condition / repair: not an order status
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({
        "Mã đơn": ["A", "A", "B", "C", "C"], "Ngày": ["01/01/2025"] * 5,
        "Trạng thái": ["Đã giao", "Đổi trả", "Đang giao", "Đã hủy", "Đã hủy"], "Tiền": [1, 2, 3, 4, 5],
    }).to_csv(d / "ban_2025.csv", index=False)
    (df,) = load_business(d).tables.values()
    assert df["status_std"].tolist() == ["returned", "returned", "shipping", "cancelled", "cancelled"]  # A: partly returned -> whole order
    assert df.attrs["status_source"] == "Trạng thái"
    # a device-condition column must NOT become status_std
    pd.DataFrame({"Máy": ["a", "b"], "Tình trạng": ["pin yếu", "đẹp 99%"]}).to_csv(d / "may.csv", index=False)
    assert "status_std" not in load_business(d).tables["may"].columns


def test_order_column_is_a_text_id_not_a_money_total():
    from datamate.data.status import find_order_col
    df = pd.DataFrame({"Tổng bill": [100.0, 100.0, 250.0], "Mã chứng từ": ["A", "B", "C"], "Số KH": ["x", "y", "z"]})
    assert find_order_col(df) == "Mã chứng từ"
    assert find_order_col(pd.DataFrame({"Tổng bill": [1.0, 2.0]})) is None
    assert find_order_col(pd.DataFrame({"Ma don": ["a"], "Mã giao dịch": ["b"]})) == "Ma don"


def test_summary_rows_are_dropped_but_surnames_survive(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({
        "Mã đơn": ["A1", "A1", None, None, "A2", None],
        "Tên hàng": ["Bia", "Mì", "TỔNG NGÀY 31/01", "Tổng tháng 1/2026", "Cộng ngày 01/02/2026", "Tong cong"],
        "Khách": ["Tong Ngoc Chi", "Tong Ngoc Chi", None, None, "Lan", None],
        "Thành tiền": [10, 20, 30, 1000, 5, 1035],
    }).to_csv(d / "ban_2026.csv", index=False)
    biz = load_business(d)
    (df,) = biz.tables.values()
    assert df["Tên hàng"].tolist() == ["Bia", "Mì"] and df["Khách"].tolist() == ["Tong Ngoc Chi"] * 2  # surname kept
    assert df["Thành tiền"].sum() == 30 and biz.notes and "4" not in biz.notes[0] or "removed" in biz.notes[0]


def test_only_adjacent_exact_duplicates_are_dropped(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({
        "Mã đơn": ["A", "A", "A", "B", "B", "B", "C", "C"],
        "Hàng": ["x", "x", "y", "x", "y", "x", "z", "w"],   # A: x,x adjacent dup -> drop one ; B: x,y,x not adjacent -> keep
        "Tiền": [5, 5, 7, 5, 7, 5, 9, 1],
    }).to_csv(d / "ban_2026.csv", index=False)
    biz = load_business(d)
    (df,) = biz.tables.values()
    assert df["Hàng"].tolist() == ["x", "y", "x", "y", "x", "z", "w"] and any("duplicate" in n for n in biz.notes)


def test_money_shorthand_with_decimal_comma_and_underscore_status(tmp_path):
    d = tmp_path / "biz"; d.mkdir()
    pd.DataFrame({
        "Ma don": ["A", "A", "B", "C"], "Ngày": ["01/01/2025"] * 4,
        "Tiền bán": ["104.000đ", "1,19tr", "1,274tr", "2tr"], "trang_thai": ["xong", "xong", "xong", "huỷ"],
    }).to_csv(d / "ban_2025.csv", index=False)
    (df,) = load_business(d).tables.values()
    assert df["Tiền bán"].tolist() == [104000, 1190000, 1274000, 2000000]
    assert df["status_std"].tolist() == ["completed", "completed", "completed", "cancelled"]  # trang_thai (underscore) recognised
