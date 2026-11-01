"""Counting fixes: raw status filters are refused once, delivery-round columns are described as pairs."""
import pandas as pd
import pytest

from datamate.data.descriptor import describe_table
from datamate.data.status import add_status_std
from datamate.sandbox.guard import GuardedExecutor


class _Inner:
    def __init__(self):
        self.ran = []

    def __call__(self, code):
        self.ran.append(code)
        return type("R", (), {"logs": "", "output": None})()


def _orders():
    df = pd.DataFrame({"Mã hóa đơn": ["A", "B", "C"], "Trạng thái": ["hủy", "Huỷ đơn", "Hoàn thành"]})
    return {"DanhSachHoaDon": add_status_std(df)}


def test_raw_status_text_filter_is_refused_once_with_the_mapping():
    ex = GuardedExecutor(_Inner())
    ex.tables = _orders()
    code = "df = tables['DanhSachHoaDon']\nn = df[df['Trạng thái'].str.contains('hủy')]['Mã hóa đơn'].nunique()\nprint(n)"
    with pytest.raises(ValueError) as e:
        ex(code)
    assert "status_std" in str(e.value) and "Huỷ đơn" in str(e.value)
    ex(code)  # second attempt runs as written
    ex2 = GuardedExecutor(_Inner())
    ex2.tables = _orders()
    ex2("df = tables['DanhSachHoaDon']\nprint((df['status_std'] == 'cancelled').sum())")
    ex2("df = tables['DanhSachHoaDon']\nprint(df['Trạng thái'].unique())")  # looking at labels is fine
    assert not ex2.status_bounced


def test_delivery_round_column_gets_a_pair_count_note():
    df = pd.DataFrame({"Mã bill": ["O1", "O1", "O1", "O2"], "Lượt giao": [1.0, 1.0, 2.0, 1.0], "Tên hàng": list("abcd")})
    txt = describe_table("SoGiaoHang", df)
    assert "(`Mã bill`, `Lượt giao`)" in txt
