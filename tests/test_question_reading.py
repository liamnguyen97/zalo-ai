"""Reading the question in code (error-analysis P0 / P1): periods (agents/periods.py), answer type and k (agents/question.py).

The sentences are the public-test questions the analysis quotes. The rulebook's dates are exact, so each case names the dates."""
from datetime import date

import pytest

from datamate.agents.explainer import code_checks, repair_spec
from datamate.agents.periods import calendar_period, covered, fix_calendar_period
from datamate.agents.question import QuestionFacts, Reading, mentioned_periods, read_question, read_type


def d(q, as_of):
    got = calendar_period(q, as_of)
    return got and (str(got[0]), str(got[1]))


def months(q, as_of):
    return [(p.label, str(p.start), str(p.end)) for p in mentioned_periods(q, as_of)]


# ------------------------------------------------------------------ A1: a year written with 2 digits
@pytest.mark.parametrize("q,as_of,expected", [
    ("Th 3/25 quan ban ra bao nhieu cai Banh croissant?", "2026-08-21", ("2025-03-01", "2025-03-31")),
    ("Lượt xe sửa điện tháng 8/25 bnhiu?", "2026-08-31", ("2025-08-01", "2025-08-31")),
    ("Trong thang 4/25 hom nao quan ban dc nhieu tien nhat z?", "2026-07-22", ("2025-04-01", "2025-04-30")),
    ("Nhom thuc pham thang 2/25 ban duoc bao nhieu tien?", "2026-07-08", ("2025-02-01", "2025-02-28")),
    ("Chị cho em hỏi th 8/25 nhóm dụng cụ y tế bán được bao nhiêu tiền ạ?", "2026-08-08", ("2025-08-01", "2025-08-31")),
    ("Th 4/25 ai mua nhieu don nhat?", "2026-05-10", ("2025-04-01", "2025-04-30")),
    ("Tháng 1/26 sp nào bán chạy nhất?", "2026-05-01", ("2026-01-01", "2026-01-31")),
    ("T3-26 doanh thu bao nhiêu", "2026-05-01", ("2026-03-01", "2026-03-31")),
    ("Quý 4/25 có bao nhiêu đơn", "2026-05-01", ("2025-10-01", "2025-12-31")),
    ("q1/26 có bao nhiêu đơn", "2026-05-01", ("2026-01-01", "2026-03-31")),
])
def test_a_two_digit_year_is_read(q, as_of, expected):
    assert d(q, as_of) == expected


@pytest.mark.parametrize("q", [
    "Th 8-15 ngày nợ bao nhiêu",                          # an age range of a debt, not 8/2015
    "công nợ 0-7, 8-15, trên 15 ngày",
    "khách nợ ngày 15/3/25 bao nhiêu",                    # a full date
    "tháng 3/12 bán được bao nhiêu",                      # 2012 is not a year of this data: the month stays this year's
])
def test_a_two_digit_number_that_is_not_a_year_is_left_alone(q):
    got = calendar_period(q, "2026-05-24")
    assert got is None or got[0].year == 2026 and not q.startswith(("công nợ", "khách"))


def test_two_digit_years_only_inside_the_window_of_the_as_of_year():
    assert d("th 3/19 bán bao nhiêu", "2026-05-24") == ("2026-03-01", "2026-03-31")      # 2019: out of the 3-year window, this year's
    assert d("th 3/23 bán bao nhiêu", "2026-05-24") == ("2023-03-01", "2023-03-31")
    assert d("th 3/27 bán bao nhiêu", "2026-05-24") == ("2026-03-01", "2026-03-31")      # after the as_of year: no


# ------------------------------------------------------------------ A2: month and year without "tháng"
@pytest.mark.parametrize("q,expected", [
    ("02/2026 có mấy khách mua từ 4 đơn trở lên?", ("2026-02-01", "2026-02-28")),
    ("6-2025 trung binh 1 don bao nhieu", ("2025-06-01", "2025-06-30")),
    ("3/26 khách đường Lý Thường Kiệt mua bao nhiêu đơn", ("2026-03-01", "2026-03-31")),
    ("07/2025 bán được bao nhiêu miếng Băng cá nhân Urgo Transparent hộp 100 miếng?", ("2025-07-01", "2025-07-31")),
])
def test_month_and_year_without_the_word_month(q, expected):
    assert d(q, "2026-05-24") == expected


@pytest.mark.parametrize("q", [
    "đơn ngày 15/03/2026 giá bao nhiêu",                  # dd/mm/yyyy
    "đơn 5/3 giá bao nhiêu",                               # day/month
    "hoá đơn 0012/2025 giá bao nhiêu",                    # a number with a slash
    "nợ đến hết ngày 04/12/2025 là bao nhiêu",
    "bán 12/10 sản phẩm",
])
def test_a_date_or_a_number_is_not_a_month(q):
    assert months(q, "2026-05-24") == []


# ------------------------------------------------------------------ A3 / A4: lists of months, chains of months
def test_a_list_of_months_shares_the_year_written_after_the_last():
    got = months("Dthu tháng 5, 6, 7/2025 lần lượt bnhiu?", "2026-05-24")
    assert got == [("month 5/2025", "2025-05-01", "2025-05-31"), ("month 6/2025", "2025-06-01", "2025-06-30"),
                   ("month 7/2025", "2025-07-01", "2025-07-31")]
    assert [m[0] for m in months("Doanh thu tháng 11/2025, 12/2025 và 1/2026 lần lượt", "2026-05-24")] == ["month 11/2025", "month 12/2025", "month 1/2026"]
    assert [m[0] for m in months("So sánh 3/25, 4/25 và 5/25 xem tháng nào bán nhiều tiền nhất", "2026-06-10")] == ["month 3/2025", "month 4/2025", "month 5/2025"]
    assert d("Dthu tháng 5, 6, 7/2025 lần lượt bnhiu?", "2026-05-24") is None                  # three periods: nothing to force


def test_the_month_before_in_a_chain_of_months_is_not_a_period():
    q = "Doanh thu tu 12/25 den 2/26 co thang sau luon cao hon thang truoc khong?"
    assert [m[0] for m in months(q, "2026-03-10")] == ["month 12/2025", "month 2/2026"]          # no "last month"
    assert [m[0] for m in months("Từ th5/2025 đến th7/2025 dthu tháng sau có luôn hơn tháng trước hông?", "2026-03-10")] == ["month 5/2025", "month 7/2025"]
    assert [m[0] for m in months("doanh thu tháng này so với tháng trước", "2026-05-24")] == ["this month", "last month"]    # a real comparison


# ------------------------------------------------------------------ A5: a year, to date
def test_a_year_and_the_year_to_date():
    assert d("doanh thu năm 2025 là bao nhiêu", "2026-05-24") == ("2025-01-01", "2025-12-31")
    assert d("doanh thu năm ngoái bao nhiêu", "2026-05-24") == ("2025-01-01", "2025-12-31")
    assert d("Từ đầu năm tới hết hnay tiệm có tổng cộng bnhiu đh rồi?", "2026-05-24") == ("2026-01-01", "2026-05-24")     # §2
    assert d("Tu dau nam toi nay co bao nhieu don roi?", "2026-05-24") == ("2026-01-01", "2026-05-24")
    assert d("từ đầu năm tới chừ bán được bao nhiêu", "2026-05-24") == ("2026-01-01", "2026-05-24")
    assert d("Từ đầu tháng này đến hết hôm nay, doanh thu bao nhiêu?", "2026-05-24") == ("2026-05-01", "2026-05-24")
    assert d("từ đầu quý này đến nay bán bao nhiêu", "2026-05-24") == ("2026-04-01", "2026-05-24")


def test_a_year_is_not_a_period_when_it_is_one_side_of_a_comparison_or_this_years_or_a_balance():
    for q in ("tháng 5/2026 so với cùng kỳ năm 2025 tăng bao nhiêu %", "so với cùng kỳ năm trước, doanh thu T3/2026",
              "doanh thu năm 2026 là bao nhiêu",               # this year: a year to date or all of it, the model's
              "tồn kho từ đầu năm đến hết hôm nay", "từ đầu năm đến hết tháng 5 doanh thu", "từ đầu tháng 5 đến hết tháng 6 doanh thu"):
        assert d(q, "2026-07-24") is None, q
    assert [m[0] for m in months("tháng 5/2026 so với cùng kỳ năm 2025 tăng bao nhiêu %", "2026-07-24")] == ["month 5/2026"]
    assert months("tồn kho từ đầu năm đến hết hôm nay", "2026-07-24") == []
    assert d("Tháng 11 năm 2025, nhóm trà có mặt hàng nào bán được nhiều ly nhất?", "2026-05-24") == ("2025-11-01", "2025-11-30")   # one month, not two periods


def test_a_forced_period_follows_the_new_readings():
    ps = [{"label": "x", "start": "2026-03-01", "end": "2026-03-31", "role": "main"}]
    notes = []
    fix_calendar_period(ps, "Th 3/25 quan ban ra bao nhieu cai Banh croissant?", "2026-08-21", notes)
    assert (ps[0]["start"], ps[0]["end"]) == ("2025-03-01", "2025-03-31") and notes


# ------------------------------------------------------------------ A6: covering a period
def test_a_period_is_covered_by_the_same_dates_or_by_the_union_of_named_periods():
    jan, dec = (date(2026, 1, 1), date(2026, 1, 31)), (date(2025, 12, 1), date(2025, 12, 31))
    q1 = [("2026-01-01", "2026-03-31")]
    assert covered(jan, q1, [jan, dec], strict=False)                            # containing it
    assert not covered(jan, q1, [jan, dec], strict=True)                         # a quarter is not the month the question names
    assert covered(jan, [("2026-01-01", "2026-01-31")], [jan, dec], strict=True)
    feb, mar = (date(2026, 2, 1), date(2026, 2, 28)), (date(2026, 3, 1), date(2026, 3, 31))
    assert covered(feb, [("2026-01-01", "2026-03-31")], [jan, feb, mar], strict=True)     # "3 tháng 1, 2, 3" as one period
    assert not covered(feb, [("2026-01-01", "2026-03-31")], [jan, feb], strict=True)       # March is not named: not a union of them


def _facts(q, as_of="2026-05-24"):
    return QuestionFacts([], read_type(q), mentioned_periods(q, as_of))


def _spec(periods, **kw):
    s, *_ = repair_spec({"rewritten_question": "q", "answer_type": "number", "periods": periods, **kw}, {}, "2026-08-31", None, None)
    return s


def test_the_check_wants_the_named_month_not_a_quarter_that_contains_it():
    q = "T1/2026 so với T12/2025 doanh thu tăng bao nhiêu %"
    quarter = _spec([{"label": "q1", "start": "2026-01-01", "end": "2026-03-31", "role": "main"},
                     {"label": "dec", "start": "2025-12-01", "end": "2025-12-31", "role": "compare"}])
    _, issues = code_checks(quarter, _facts(q, "2026-05-24"), q)
    assert any("month 1/2026" in i and "equals it" in i for i in issues) and len(issues) == 1
    exact = _spec([{"label": "jan", "start": "2026-01-01", "end": "2026-01-31", "role": "main"},
                   {"label": "dec", "start": "2025-12-01", "end": "2025-12-31", "role": "compare"}])
    assert code_checks(exact, _facts(q, "2026-05-24"), q)[1] == []


def test_a_balance_or_a_range_may_have_a_spec_period_wider_than_the_phrase():
    q = "tồn kho đến hết tháng 5/2026 là bao nhiêu"
    wide = _spec([{"label": "all", "start": "2025-01-01", "end": "2026-05-31", "role": "main"}])
    assert code_checks(wide, _facts(q), q)[1] == []
    q = "doanh thu từ tháng 3/2026 đến tháng 5/2026"
    span = _spec([{"label": "r", "start": "2026-03-01", "end": "2026-05-31", "role": "main"}])
    assert code_checks(span, _facts(q), q)[1] == []


# ------------------------------------------------------------------ C1 / C2: k
@pytest.mark.parametrize("q,k,kind", [
    ("Tháng 1/26 sp nào bán chạy nhất?", None, "string"),
    ("Mặt hàng nào bán chạy nhất 30 ngày qua?", None, "string"),
    ("Mặt hàng nào bán chạy nhất 7 ngày gần nhất?", None, "string"),
    ("Kể giúp các sp còn lô tồn hết hạn trong 90 ngày tới", None, None),
    ("Công nợ theo nhóm tuổi nợ 0–7 ngày, 8–15 ngày, trên 15 ngày là bao nhiêu mỗi nhóm", None, None),
    ("Công nợ theo nhóm tuổi nợ 0-7, 8-15, trên 15 ngày lần lượt là bao nhiêu", None, None),
    ("Mặt hàng nào bán chạy nhất trong 7 ngày?", None, "string"),
    ("Top 3 sp bán nhiều nhất tháng 3 năm 2026 là những gì?", 3, "list"),
    ("3 ngày bán nhiều nhất tháng 5/2026 là những ngày nào?", 3, None),                 # a real top 3 days
    ("Doanh thu 3 tháng 7/2025, 8/2025, 9/2025 lần lượt là bao nhiêu", 3, "list"),       # one number per month
    ("dthu 3 tháng liền nhau kết thúc ở tháng 12/2025 là bao nhiêu mỗi tháng", 3, "list"),
])
def test_k_is_the_number_of_elements_not_a_window_of_time(q, k, kind):
    r = read_type(q)
    assert r.k == k, q
    if kind:
        assert r.answer_type == kind, q


def test_k_overrides_the_spec_only_when_the_reading_is_sure():
    q = "tháng 5 bán được 3 sp"
    facts = _facts(q)
    facts.reading = Reading("number", 3, "default", False)
    s = _spec([], answer_type="list", k=5)
    assert code_checks(s, facts, q)[0] == [] and s.k == 5                         # a default reading never touches k
    facts.reading = Reading("list", 3, "which, several", True)
    code_checks(s, facts, q)
    assert s.k == 3


# ------------------------------------------------------------------ C4 / C5: table and count
@pytest.mark.parametrize("q", [
    "cho hỏi tính đến hết ngày 12/07/2025, 5 khách nào đang nợ nhiều nhất, mỗi người còn nợ bao nhiu",
    "tinh den het ngay 14/09/2026, 5 khach nao dang no nhiu nhat, moi nguoi con no bnhiu ha",
    "Cho mình bảng 3 khách còn nợ nhiều nhất tính đến hết ngày 05/08/2026, kèm số tiền còn nợ.",
    "cho bang 5 khach con no nhieu nhat tinh den het ngay 27/03/2026, kem so tien con no",
    "liet ke 3 khach no nhieu nhat tinh den het ngay 28/06/2026, kem so tien con no",
    "Liệt kê 3 khách nợ nhiều nhất tính đến hết ngày 19/08/2026, kèm số tiền còn nợ.",
    "Bảng hoa hồng thợ th7/2026, ai nhiều tiền thì xếp lên trên nghe.",
    "Hoa hồng từng thợ th 3/26, cho bảng xếp tiền giảm dần nghe?",
    "e ơi cko a xin cái bảng tien hoa hồng kua mọi thợ th11/2025 nha, xep tu cao xuong thap",
])
def test_a_table_is_read_from_the_wording(q):
    r = read_type(q)
    assert (r.answer_type, r.sure) == ("table", True), q


@pytest.mark.parametrize("q", [
    "dthu các đơn trả bằng chuyển khoản tháng 9/2025 là bao nhiêu?",              # "bằng" is not "bảng"
    "cho hoi thang 5/2026, dthu cac don tra bang chuyen khoan la bao nhieu",
    "Tháng 1/2026 ai là khách mua nhiều lần nhất, tính theo số đơn?",
    "Top 3 khách hàng có công nợ lớn nhất đến hết ngày 28/07/2026 và số nợ của từng người?",       # (a table, by the older rule)
])
def test_words_that_look_like_a_table_but_are_not(q):
    r = read_type(q)
    assert r.answer_type != "table" or "và số nợ" in q


@pytest.mark.parametrize("q,kind", [
    ("Tháng 4/2026 có bao nhiêu đơn từ 190k trở lên?", "number"),
    ("Th5 mấy khách mua từ 9 đơn", "number"),
    ("tháng 5 bán được bao nhiêu ly", "number"),
    ("Có bao nhiêu mặt hàng nằm trong ít nhất 122 đơn", "number"),
])
def test_a_count_of_objects_is_a_number_for_sure(q, kind):
    r = read_type(q)
    assert (r.answer_type, r.sure) == (kind, True), q


@pytest.mark.parametrize("q", [
    "Th 4/25 ai mua nhieu don nhat?",                     # who bought the most orders: a name
    "Khách nào mua nhiều đơn hàng nhất tháng 7/2025?",
    "cuoi ngay hom nay da 4x6 con may xe 4m3 chan, may m3 le",          # two stock quantities
    "Cát xay to còn bao nhiêu xe 4m3 chẵn, lẻ bao nhiêu m3",
])
def test_a_superlative_or_two_quantities_are_not_a_single_count(q):
    r = read_type(q)
    assert not (r.rule == "how many"), q


def test_read_question_gets_the_new_periods():
    f = read_question("Th 3/25 quan ban ra bao nhieu cai Banh croissant?", "2026-08-21", {})
    assert [(p.label, str(p.start)) for p in f.periods] == [("month 3/2025", "2025-03-01")]


# ------------------------------------------------------------------ text.norm: telex slips, the lexicon (tickets T3.2 / T3.4)
def test_a_tone_letter_left_on_a_long_word_is_undone_only_when_a_syllable_is_left():
    from datamate.text import norm
    assert norm("thangs 6 năm 2025") == "thang 6 nam 2025" and norm("khachs nao") == "khach nao"
    assert norm("nhieuf") == "nhieu" and norm("thoaij") == "thoai"
    assert norm("thanks") == "thanks" and norm("class") == "class"           # what is left is no Vietnamese syllable


def test_the_lexicon_writes_each_spelling_in_its_standard_form():
    from datamate.text import norm
    assert norm("Quý 4 năm 2025 tiệm chốt được bao nhiêu o đờ hả?") == "quy 4 nam 2025 tiem chot duoc bao nhieu don hang ha"
    assert norm("Từ đầu năm tới hết hnay tiệm có tổng cộng bnhiu đh rồi?") == "tu dau nam toi het hom nay tiem co tong cong bao nhieu don hang roi"
    assert norm("tháng 5 bán đc nhiêu ly") == "thang 5 ban duoc bao nhieu ly"     # "nhiêu" alone, with its circumflex
    assert norm("bán được bao nhiêu ly") == "ban duoc bao nhieu ly"              # already the standard: no "bao bao"
    assert norm("ai mua nhiều đơn nhất") == "ai mua nhieu don nhat"              # "nhiều" is not "nhiêu"
    assert norm("trả bằng ck chiếm bn %") == "tra bang chuyen khoan chiem bao nhieu %"
    assert norm("cko a xin cái bảng kua mọi thợ") == "cho a xin cai bang cua moi tho"
    # left alone on purpose: folded "o do" is also "ở đó", "đt" is "điện thoại" in a phone shop, "HD" is part of a product name
    assert norm("ở đó bán bao nhiêu") == "o do ban bao nhieu"
    assert norm("máy đt") == "may dt" and norm("Telfast HD 180mg") == "telfast hd 180mg" and norm("SP0003") == "sp0003"


def test_the_lexicon_can_be_switched_off(monkeypatch):
    from datamate import lexicon
    from datamate.text import norm
    monkeypatch.setattr(lexicon, "ENABLED", False)
    assert norm("bnhiu đh") == "bnhiu dh"


def test_a_superlative_of_a_count_is_a_name_not_a_count():
    for q in ("Th 9/25 ai là khách mua nhiêuf lần nhất, đếm theo số đơn?", "Khách nào mua nhiều đơn hàng nhất tháng 7/2025?",
              "khach nao mua nhiu don hang nhat thang 4/2025"):
        r = read_type(q)
        assert (r.answer_type, r.rule) == ("string", "which, one"), q
    for q in ("Cho hỏi bao nhiêu khách mua ít nhất 6 đơn trong tháng 8 năm 2025?", "Trong tháng 1/2026, bao nhiêu xe Air Blade khác nhau vào tiệm?"):
        assert read_type(q).rule == "how many", q


def test_the_spec_prompt_explains_the_shop_owners_words_it_uses(monkeypatch):
    """public q_37a4aca072a6 "chốt được bao nhiêu o đờ" was read as revenue: the model reads the question as written, not norm's output."""
    from datamate import lexicon
    f = read_question("Quý 4 năm 2025 tiệm chốt được bao nhiêu o đờ hả?", "2026-06-24", {})
    assert f.glossary == [("o đờ", "đơn hàng (order)")] and '"o đờ" = đơn hàng (order)' in f.prompt() and f.as_log()["glossary"]
    two = read_question("Tuần trước mỗi đh mang về bao nhiêu, trả bằng ck?", "2026-06-24", {}).glossary
    assert two == [("đh", "đơn hàng (order)"), ("ck", "chuyển khoản (bank transfer)")]
    plain = read_question("doanh thu tháng 5 bao nhiêu", "2026-06-24", {})
    assert plain.glossary == [] and "WORDS OF THE QUESTION" not in plain.prompt() and "glossary" not in plain.as_log()
    monkeypatch.setattr(lexicon, "GLOSSARY", False)
    assert read_question("bao nhiêu o đờ", "2026-06-24", {}).glossary == []


def test_dialect_words_and_protected_phrases():
    from datamate.text import norm
    assert norm("Hom ni quan duoc may don roi?") == "hom nay quan duoc may don roi"                 # central "ni" = "này"
    assert norm("2/2026 hàng mô lãi gộp nhiều nhất rứa?") == "2/2026 hang nao lai gop nhieu nhat vay"
    assert norm("trog th11/2025 nv nào bán đc dthu cao nhứt") == "trong th11/2025 nhan vien nao ban duoc doanh thu cao nhat"
    assert norm("sửa xe mô tô hết bao nhiêu") == "sua xe mo to het bao nhieu"                      # "mô tô" is protected
    assert norm("dthu các đơn trả xi ô đi") == "doanh thu cac don tra cod"                          # COD spelled out
    assert read_type("Th 3/2026 thợ mô làm nhiều lượt nhất?").answer_type == "string"


def test_two_groups_and_each_of_them_is_one_number_per_group():
    """C6 (ticket T4.2)."""
    for q in ("tháng này, dthu các đơn trả bằng cod và các đơn trả bằng tm mỗi loại bao nhiêu",
              "Dthu tháng 5/2026 qua Zalo OA và qua điện thoại, mỗi kênh bnhiu?"):
        r = read_type(q)
        assert (r.answer_type, r.rule, r.sure) == ("list", "one number per group", True), q
    assert read_type("trong thang nay, so ly size m ban dc la bao nhieu, tinh moi loai do uong").answer_type == "number"   # one total


def test_a_count_in_the_shops_own_units_is_a_number():
    """T4.1: the units come from the shop's unit columns, not from a list in the code."""
    import pandas as pd
    t = {"ban": pd.DataFrame({"ĐVT": ["kg", "bao", "Đĩa", "Thùng 24 lon", None], "Tên hàng": ["a", "b", "c", "d", "e"]})}
    for q in ("tháng 5 bán được bao nhiêu kg nếp", "tháng 5 bán được mấy đĩa"):
        assert read_question(q, "2026-06-01", t).reading.rule == "how many", q
    assert read_question("tháng 5 bán được mấy đĩa", "2026-06-01", {}).reading.rule == "default"      # another shop: not its unit
    assert read_question("bao nhiêu kg và bao nhiêu bao", "2026-06-01", t).reading.rule == "default"   # two quantities asked


def test_two_words_inside_a_longer_value_are_not_a_name_and_header_rows_are_no_values():
    """B5: 29 of 30 two-word matches inside a value were wrong (people's names, status labels)."""
    import pandas as pd
    from datamate.agents.question import find_names
    t = {"kh": pd.DataFrame({"Tên khách hàng": ["Nghiêm Bình Quân", "Triệu Trung Bình", "Tên khách hàng", "Lê Văn An"]}),
         "nhap": pd.DataFrame({"Trạng thái": ["Đã nhập hàng", "Đã hủy"] * 2}),
         "hang": pd.DataFrame({"Tên hàng": ["Đậu phộng rang muối 200g", "Bánh tiramisu", "Dây đeo điện thoại", "Trà sữa"]})}
    names = lambda q: [n.phrase for n in find_names(q, t)]                                   # noqa: E731
    assert names("doanh thu bình quân mỗi đơn tháng 5") == []                                # not the customer "Nghiêm Bình Quân"
    assert names("trung bình một đơn bao nhiêu") == [] and names("tổng tiền nhập hàng tháng 5") == []
    assert names("đơn bán qua điện thoại tháng 5") == []                                     # not "Dây đeo điện thoại"
    assert names("khách hàng nào mua nhiều nhất") == []                                      # "Tên khách hàng" is a header row
    assert names("rang muối 200g bán được bao nhiêu") == ["rang muoi 200g"]                  # three words: the product
    assert names("bánh tiramisu bán được bao nhiêu") == ["banh tiramisu"]                    # a whole two-word value: exact
