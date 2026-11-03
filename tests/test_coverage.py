"""§7 information coverage (agents/coverage.missing_information): kinds of information a shop may not record."""
from types import SimpleNamespace

import pandas as pd

from datamate.agents.coverage import coverage_fact, missing_information


def _shop(tables, readme="", orders=None, **engines):
    return SimpleNamespace(tables=tables, readme=readme, orders=orders, debt=engines.get("debt"), purchases=engines.get("purchases"))


SALES = {"ban_hang": pd.DataFrame({"Ngày": pd.to_datetime(["2026-01-01"]), "Khách hàng": ["Lê Văn Nam"], "Tên hàng": ["Keo dán ống PVC"],
                                   "Thành tiền": [1000.0]})}


def names(q, shop):
    return [n.name for n in missing_information(q, shop)]


def test_a_kind_of_information_the_shop_does_not_record():
    shop = _shop(SALES)
    assert names("e thang 1/2026 khach nu mua tong cong bao nhieu tien", shop) == ["gender"]
    assert names("Có bao nhiêu khách hàng ở đường Trần Não?", shop) == ["address"]
    assert names("Cho hỏi tổng tiền nhập hàng tháng này là bao nhiêu?", shop) == ["purchase"]
    assert names("het hom nay bia tiger con lai may lon trong kho", shop) == ["stock"]
    assert names("Đến hết ngày 09/07/2026, số tiền khách còn thiếu là bao nhiêu?", shop) == ["debt"]
    assert "coverage" not in coverage_fact([]) and "records no customers' gender" in coverage_fact(missing_information("khách nữ", shop))


def test_not_asked_or_recorded():
    shop = _shop(SALES)
    assert names("check gium thang 5/2025 ban ra bao nhiu lon keo dan ong pvc binh minh 200g", shop) == []   # "dán ống", not a gender
    assert names("khách Giang Phương Thu mua chịu thêm bao nhiêu", _shop({**SALES, "no": pd.DataFrame({"Còn nợ": [1.0]})})) == []
    assert names("đang tính nhập thêm máy cũ, tháng này bán ra được bao nhiêu điện thoại cũ", shop) == []   # a sales question
    with_gender = _shop({**SALES, "khach": pd.DataFrame({"Tên": ["A", "B"], "Phân loại": ["Nam", "Nữ"]})})
    assert names("khách nữ mua bao nhiêu tiền", with_gender) == []                        # gender labels, whatever the column name
    documented = _shop(SALES, readme="## Cột trong `ban_hang.csv`\n- **DiaChiGiao**: địa chỉ giao hàng\n")
    assert names("khách ở đường Lê Lợi mua bao nhiêu", _shop({"t": pd.DataFrame({"DiaChiGiao": ["1 Lê Lợi"]})})) == []   # CamelCase
    assert names("khách ở đường Lê Lợi mua bao nhiêu", documented) == []                  # a README column description


def test_customers_named_by_no_order():
    walk_in = pd.DataFrame({"order": ["1", "2"], "customer_key": ["", ""]})
    named = pd.DataFrame({"order": ["1", "2"], "customer_key": ["", "le van nam"]})
    q = "bn khach mua it nhat 14 don trong thang 12 nam 2025"
    assert names(q, _shop(SALES, orders=walk_in)) == ["customers"]
    assert names(q, _shop(SALES, orders=named)) == []
    assert names(q, _shop(SALES, orders=None)) == []                                      # no sales_orders: not judged
    assert names("doanh thu từ khách lẻ tháng 1", _shop(SALES, orders=walk_in)) == []
    assert names("tiệm có bnhiu xe khách quen", _shop(SALES, orders=walk_in)) == []      # regulars by plate (repair shops)


Q_GENDER = "e thang 5/2026 khach nu mua tong cong bao nhieu tien"


def test_gate_modes():
    from tests.test_orchestrator import RIGHT, SPEC, Scripted, _orch, _st
    # fact (the default): the question runs as usual, the Coder is told the shop records no gender
    m = Scripted(code=[RIGHT], explain=[{"spec": SPEC}])
    st = _st(Q_GENDER)
    _orch(m).run(st)
    assert m.prompts("explain") and "records no customers' gender" in m.prompts("code")[0]
    assert any(e.get("coverage_needs") == ["gender"] and e.get("mode") == "fact" for e in st.log["orchestrator"])
    # on: UNANSWERABLE in code, before any model call
    m_on = Scripted(code=[], explain=[{"spec": SPEC}])
    out = _orch(m_on, coverage_gate="on").run(_st(Q_GENDER))
    assert out["type"] == "unanswerable" and "gender" in out["reason"] and not m_on.prompts("explain")
    # off: not run, nothing said
    m_off = Scripted(code=[RIGHT], explain=[{"spec": SPEC}])
    st_off = _st(Q_GENDER)
    _orch(m_off, coverage_gate="off").run(st_off)
    assert "records no" not in m_off.prompts("code")[0] and not any("coverage_needs" in e for e in st_off.log["orchestrator"])


def test_missing_fields_named_after_a_column_are_not_missing():
    from datamate.agents.explainer import recorded_column
    tables = {"khach": pd.DataFrame({"GioiTinh": ["Nam"], "Tên": ["A"]}), "don": pd.DataFrame({"Kênh bán hàng": ["Zalo"]})}
    assert recorded_column("giới tính", tables) == "khach`.`GioiTinh"
    assert recorded_column("kênh bán", tables) == "don`.`Kênh bán hàng"
    assert recorded_column("địa chỉ", tables) is None and recorded_column("nữ", tables) is None   # absent; too short to tell


def test_a_period_partly_inside_the_data_is_said_so():
    from datamate.agents.facts import verified_facts
    so = pd.DataFrame({"date": pd.to_datetime(["2025-03-05", "2025-03-20", "2025-04-02"]), "valid": [True] * 3,
                       "status_std": ["completed"] * 3})
    v = {"sales_orders": so, "period_start": pd.Timestamp("2025-01-01"), "period_end": pd.Timestamp("2025-03-31 23:59")}
    text = verified_facts("doanh thu từ đầu năm tới cuối tháng 3", None, v)
    assert "only part of this period has data" in text and "NOT unanswerable" in text
    v2 = {**v, "period_start": pd.Timestamp("2025-03-10"), "period_end": pd.Timestamp("2025-03-31 23:59")}
    assert "only part" not in verified_facts("doanh thu tháng 3", None, v2)


def test_a_surname_or_the_year_is_not_a_street_or_a_gender():
    named = pd.DataFrame({"order": ["1", "2"], "customer_key": ["duong minh ha", "x"], "customer": ["Dương Minh Hà", "Lê Văn Nam"]})
    shop = _shop(SALES, orders=named)
    assert names("khách Dương Minh Hà mua bao nhiêu tiền tháng 3", shop) == []            # "Dương" the surname
    assert names("khach duong minh ha mua bao nhieu tien thang 3", shop) == []            # no accents: a customer's name
    assert names("khách đường Lý Thường Kiệt mua bao nhiêu", shop) == ["address"]         # "đường" written with its accents
    assert names("khách năm nay mua bao nhiêu", shop) == [] and names("số khách năm 2025", shop) == []
    assert names("khach nam nay mua bao nhieu", shop) == []                               # the year, typed without accents
    assert names("doanh thu từ khách nam năm nay", shop) == ["gender"]
