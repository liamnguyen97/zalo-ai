"""Table structure measured on the data (data/structure.py) and where it is used: cards, table names, misplaced columns."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamate.agents.explainer import repair_spec
from datamate.agents.lookups import DataTools
from datamate.data.descriptor import describe_business, describe_table
from datamate.data.loader import Business
from datamate.data.structure import (card_lines, detect_structure, find_doc_level, find_links, find_totals, readme_notes,
                                     resolve_table)


def _purchase_shop(n_slips=60, n_items=40, seed=0):
    """A purchase slip table (one row per slip, total = sum qty x price), its lines, a product master and the sales lines."""
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n_slips):
        for j in range(int(rng.integers(2, 6))):
            lines.append({"Mã nhập hàng": f"PN{i:05d}", "Mã hàng": f"SP{int(rng.integers(n_items)):04d}",
                          "Số lượng": float(rng.integers(1, 20)), "Đơn giá": float(rng.integers(10, 300)) * 1000})
    L = pd.DataFrame(lines)
    tot = (L["Số lượng"] * L["Đơn giá"]).groupby(L["Mã nhập hàng"]).sum()
    H = pd.DataFrame({"Mã nhập hàng": tot.index, "Thời gian": pd.date_range("2025-01-01", periods=len(tot), freq="D"),
                      "Cần trả NCC": tot.values, "Giảm giá": 0.0})
    P = pd.DataFrame({"Mã hàng": [f"SP{k:04d}" for k in range(n_items)], "Tên hàng": [f"Hàng {k}" for k in range(n_items)]})
    sales = pd.DataFrame({"Mã hóa đơn": [f"HD{k:06d}" for k in range(n_items * 80)],
                          "Mã hàng": [f"SP{k % n_items:04d}" for k in range(n_items * 80)], "Đơn giá": 5000.0})
    return {"File__DanhSachNhapHang": H, "File__DanhSachChiTietNhapHang": L, "File__DanhSachSanPham": P,
            "DanhSachHoaDonChiTiet": sales}


def test_parent_child_link_is_found_but_not_a_master_over_many_rows():
    T = _purchase_shop()
    links = {(l.head, l.lines): l for l in find_links(T)}
    slip = links[("File__DanhSachNhapHang", "File__DanhSachChiTietNhapHang")]
    assert slip.key == "Mã nhập hàng" and "Thời gian" in slip.head_only and "Cần trả NCC" in slip.head_only
    # 40 products over 3,200 sales lines = 80 rows per product: a master, not a parent with a few child rows
    assert ("File__DanhSachSanPham", "DanhSachHoaDonChiTiet") not in links


def test_document_total_is_found_and_zero_equals_zero_proves_nothing():
    T = _purchase_shop()
    totals = find_totals(T, find_links(T))
    by_col = {(t.table, t.column): t for t in totals}
    t = by_col[("File__DanhSachNhapHang", "Cần trả NCC")]
    assert t.expr == "`Số lượng` × `Đơn giá`" and t.share == 1.0
    # `Giảm giá` is 0 on every slip and some line product is never 0: no "total" claim from 0 = 0 rows
    assert ("File__DanhSachNhapHang", "Giảm giá") not in by_col


def test_flat_table_marks_per_document_columns_not_a_mostly_zero_line_column():
    rows = []
    for i in range(80):
        day = pd.Timestamp("2025-03-01") + pd.Timedelta(days=i % 30)
        for j in range(3):
            rows.append({"Giờ": f"{day:%Y-%m-%d} 08:{i % 60:02d}:00", "Mã hóa đơn": f"HD{i:04d}", "Ngày": day,
                         "Khách": f"Khách {i % 7}", "Hàng": f"H{j}", "SL": j + 1,
                         "SL trả": (1 if (i % 4 == 0 and j == 0) else 0)})
    flat = pd.DataFrame(rows)
    doc, cols = find_doc_level({"ban_hang": flat}, [])["ban_hang"]
    assert doc == "Mã hóa đơn"                      # not the time-as-text column, which repeats the same way
    assert {"Ngày", "Khách"} <= set(cols) and "SL trả" not in cols and "Hàng" not in cols


def test_readme_notes_go_to_the_table_the_sentence_is_about():
    T = _purchase_shop()
    readme = ("## Quy ước\n- **Nhập hàng**: **Đơn giá** là giá nhập một đơn vị, theo đơn vị nhập của dòng; "
              "**Cần trả NCC** là tổng tiền của cả phiếu nhập.\n- **Ghi chú**: `DanhSachSanPham.Tên hàng` là tên in trên tem.\n")
    notes = readme_notes(readme, T, find_links(T))
    # `Đơn giá` is in the purchase lines AND the sales lines: the slip link to `Cần trả NCC` decides
    assert "giá nhập một đơn vị" in notes["File__DanhSachChiTietNhapHang"]["Đơn giá"]
    assert "Đơn giá" not in notes.get("DanhSachHoaDonChiTiet", {})
    assert "tổng tiền của cả phiếu" in notes["File__DanhSachNhapHang"]["Cần trả NCC"]
    assert "tem" in notes["File__DanhSachSanPham"]["Tên hàng"]


def test_table_names_from_the_readme_resolve_to_the_loader_key():
    T = _purchase_shop()
    assert resolve_table("DanhSachNhapHang", T) == "File__DanhSachNhapHang"
    assert resolve_table("DanhSachChiTietNhapHang.xlsx", T) == "File__DanhSachChiTietNhapHang"
    assert resolve_table("File", T) is None                      # the workbook holds several sheets: ambiguous
    assert resolve_table("KhongCo", T) is None


def test_cards_carry_the_structure_and_stay_unchanged_without_it():
    T = _purchase_shop()
    st = detect_structure(T, "Các danh sách `DanhSachNhapHang`, `DanhSachChiTietNhapHang` nằm chung file.")
    head = describe_table("File__DanhSachNhapHang", T["File__DanhSachNhapHang"], structure=st)
    lines = describe_table("File__DanhSachChiTietNhapHang", T["File__DanhSachChiTietNhapHang"], structure=st)
    assert "PARENT of `File__DanhSachChiTietNhapHang`" in head and "DOCUMENT TOTAL" in head and "README name: `DanhSachNhapHang`" in head
    assert "CHILD of `File__DanhSachNhapHang`" in lines and "only the parent has `Thời gian`" in lines
    assert describe_table("File__DanhSachNhapHang", T["File__DanhSachNhapHang"]) == describe_table(
        "File__DanhSachNhapHang", T["File__DanhSachNhapHang"], structure=None)
    assert card_lines("x", None) == ([], {})
    cards = describe_business(Business("b", "", T), st)
    assert "`File__DanhSachNhapHang`.`Mã nhập hàng` ↔ `File__DanhSachChiTietNhapHang`" not in cards.get("__relations__", "")


def test_a_column_put_on_the_lines_table_moves_to_its_parent():
    # train q_00a6e23cbacf: the spec put the slip date `Thời gian` on the slip LINES; it stayed an error for the rest of the loop
    T = _purchase_shop()
    tools = DataTools(T, None, detect_structure(T))
    spec = {"rewritten_question": "avg price", "answer_type": "number", "measure": "money",
            "tables": {"DanhSachChiTietNhapHang": ["Mã nhập hàng", "Thời gian", "Số lượng", "Đơn giá"]}}
    s, errs, warns, notes = repair_spec(spec, T, "2025-03-31", tools, "giá nhập bình quân")
    assert errs == []
    assert s.tables["File__DanhSachChiTietNhapHang"] == ["Mã nhập hàng", "Số lượng", "Đơn giá"]
    assert s.tables["File__DanhSachNhapHang"] == ["Thời gian"]
    assert any("`Thời gian` is not a column" in n and "join" in n for n in notes)


def test_lookups_accept_readme_names_and_search_other_tables():
    T = _purchase_shop()
    tools = DataTools(T)
    assert "is the table `File__DanhSachNhapHang`" in tools.describe("DanhSachNhapHang")
    out = tools.find_values("Hàng 7", table="File__DanhSachChiTietNhapHang")   # the lines hold codes, the name is in the master
    assert "found in other tables" in out and "File__DanhSachSanPham" in out


def test_sandbox_maps_a_readme_table_name():
    from datamate.sandbox.guard import GuardedExecutor

    class Inner:
        def __call__(self, code):
            self.code = code
            return "ran"

    inner = Inner()
    g = GuardedExecutor(inner)
    g.tables = _purchase_shop()
    g("df = tables['DanhSachNhapHang']\nprint(len(df))")
    assert "File__DanhSachNhapHang" in inner.code
    # a case-only slip was "fixed" on the parsed tree but the original text was run (the fix never reached the sandbox)
    g("df = tables['file__danhsachnhaphang']\nprint(len(df))")
    assert "tables['File__DanhSachNhapHang']" in inner.code


TRAIN = Path(__file__).resolve().parents[1] / "train" / "data" / "biz_109"


@pytest.mark.skipif(not TRAIN.exists() or os.getenv("DATAMATE_DATA_TESTS") != "1",
                    reason="reads a real shop (~10 s): set DATAMATE_DATA_TESTS=1 with train/ unpacked")
def test_real_shop_purchase_slip_and_its_total():
    from datamate.data.loader import load_business
    biz = load_business(TRAIN)
    st = detect_structure(biz.tables, biz.readme)
    assert any(l.head == "DuLieuBanHang__DanhSachNhapHang" and l.lines == "DuLieuBanHang__DanhSachChiTietNhapHang" for l in st.links)
    assert any(t.column == "Cần trả NCC" and t.expr == "`Số lượng` × `Đơn giá`" for t in st.totals)
    assert st.aliases["DanhSachNhapHang"] == "DuLieuBanHang__DanhSachNhapHang"
    assert "giá nhập" in st.notes["DuLieuBanHang__DanhSachChiTietNhapHang"]["Đơn giá"]
