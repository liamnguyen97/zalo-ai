# Tickets: tăng accuracy bằng cơ chế chung (sau error analysis dev300 run `2b4ca48e19`)

Nguồn: hai lượt đánh giá đề xuất ngày 2026-11-03 (engine check, judge, lexicon). Mọi số đo trên log dev300 của run `20261102-153007_dev300`
(PA 0,760). Các luật dưới được thiết kế khi nhìn chính log này, nên số "trước khi chạy model" là **trần**, cần một lượt dev300 / public để xác nhận.

Nguyên tắc: không logic theo business id hay theo câu; mỗi luật phải là luật trên chữ ký hàm, kiểu đáp án, dữ liệu của shop hoặc ngữ pháp; edge case là
một dòng dữ liệu (từ điển / bảng precision), không phải một nhánh code. Mỗi ticket có test, giữ toàn bộ test cũ xanh, ghi công tắc `DATAMATE_*` vào
run meta khi có hành vi mới.

Trạng thái: `TODO` · `DOING` · `DONE` · `BACKLOG`

| ID | Việc | Trạng thái | Trần dev300 |
|---|---|---|---|
| T1.1 | Làm sạch `engine_call` thay vì bỏ cả call | DONE | 4 câu (59/59 ref đúng, replay code thật) |
| T1.2 | `route()` bỏ entity là đơn vị cơ bản của mặt hàng | DONE | 1 câu (`q_1885`) |
| T1.3 | Ép engine hint theo bảng precision của từng hàm | DONE | 3 câu (sau T1.1/T1.2) |
| T2 | Judge: trả lời bị cắt ở `max_tokens` | DONE | ~0 (bỏ một lỗi im lặng) |
| T3.1 | Script khám phá từ lạ + log từ lạ mỗi câu | DONE | 0 (công cụ) |
| T3.2 | Lexicon duyệt tay trong `norm` | DONE | train reading 0,9590 → 0,9614 |
| T3.3 | Dòng chú giải lexicon vào prompt spec (C7) | DONE | 0 dev300; 2 câu public `o đờ` |
| T3.4 | Luật telex có kiểm âm tiết, thay `_THANG` | DONE | 0 (tổng quát hoá một vá) |
| T4.1 | Đơn vị lấy từ dữ liệu cho luật đếm | DONE | 0 đổi kiểu; 143 câu thành "chắc chắn" |
| T4.2 | C6: "mỗi <nhóm> bao nhiêu" → nhiều số | DONE | train 7 sửa / 0 hỏng |
| B1 | Judge: nới bộ lọc evidence (mảnh khớp FACTS) | DONE | 2–3 revise đúng trên câu sai |
| B2 | Nhãn thanh toán lấy từ dữ liệu cho `status` | DONE | 0/44 shop đổi |
| B4 | Judge: `verdict, target, evidence` đầu JSON; parse JSON bị cắt | DONE | chưa đo (prompt) |
| D1 | Gộp lexicon thành một dict chung + phương ngữ | DONE | public 26 câu đọc đúng hơn |
| B5 | `find_names`: khớp "inside one value" quá rộng | DONE | 83 tên rác bị bỏ |
| B6 | `debt.METHOD_COL` thiếu `Thanh toán bằng` (dev_008) | DONE | 0/10 shop công nợ đổi |
| B7 | Công nợ ghi trong ghi chú đơn (dev_008: không có phiếu thu) | DONE (P3.1b) | 0 dev300 (không câu nợ ở các shop này) |
| P2.1 | Mã đơn dạng STT, "mỗi dòng một phiếu", điền ô cấp đơn | DONE | 0 dev300; public dev_011/012/014 (153 câu) có `sales_orders`, trusted |
| P2.2 | Bảng viết tắt → tên đầy đủ + category (theo hình dạng) | DONE | 0 dev300; public dev_011/014 (103 câu) |
| P2.5' | Cổng "ref cùng loại với đáp án" (thay abstain khi `engine_mismatch`) | DONE | replay 3 run dev300: bỏ 17 ref sai, 0 ref đúng |
| P2.3' | Gate §7 "shop không ghi loại thông tin này" (thay phần từ điển tính chất) | DONE | train 28 lần bắn, 28 gold unanswerable, 0 sai |
| P3.1a | Tên người: bỏ kính ngữ, khớp tên gọi với danh sách (duy nhất) | DONE | 0 dev300; public dev_012 |
| P2.4 | `missing_fields` trùng tên cột bị bỏ; fact "kỳ chỉ phủ một phần" | DONE | ~0 (1 spec dev300 khai, khai đúng) |
| P3.1c | Engine lô / hạn dùng (§6) | DONE | train 10/10 câu hạn dùng khớp gold qua hàm |
| B3 | Route mới: `debt_by_customer`, xếp hạng khách theo số đơn; engine `customers(...)` | BACKLOG | ~5 câu (báo cáo dev300) |

---

## T1. Kiểm bằng giá trị engine (gộp đề xuất 1 và 2)

Bối cảnh: cụm lỗi lớn nhất của dev300 là "engine đã có số đúng nhưng Coder không dùng". `build_reference` bỏ cả call khi có một tham số lạ
(`inspect.signature(fn).bind`), không log gì: 120/242 `engine_call` bị bỏ, 64 vì `sales_metrics(metric=...)`. Hint của `route()` chỉ được ghi log
(`orchestrator.py`, "measured, not enforced"). `agrees()` coi mọi dict là "đồng ý" nên ref `sales_metrics` không chặn được gì.

### T1.1 Làm sạch `engine_call` — DONE
- Bỏ tham số hình thức: `valid_only=True`; `metric` có trong danh sách trắng thì đổi thành key của dict kết quả (`order_count→orders`,
  `avg_order_value→avg_order`, `revenue→revenue`…); `unit` chỉ bỏ khi **bằng đơn vị cơ bản** của mặt hàng.
- Metric lạ (vd `cancelled_orders`), tham số điều kiện (`shift`, `channel`, `group`, `items` list…) → **vẫn bỏ cả call** (bỏ đi thì hàm trả số
  toàn shop, mâu thuẫn với đáp án đúng).
- Call đã phải làm sạch chỉ được dùng khi spec hỏi `number` (một doanh thu không phải đáp án của câu hỏi phần trăm).
- Mỗi call bị bỏ ghi lý do vào `log["orchestrator"]` (`engine_call_dropped`).
- Bằng chứng: đo lỏng (đúng như đề xuất gốc) thì 4 câu được / 14 câu hại; với 3 luật trên: 58 call khôi phục, 58/58 đúng gold, 4 câu đang sai
  (`q_2d2b`, `q_e94f`, `q_4362`, `q_6292`), 0 câu đúng bị đẩy sai.
- Xong khi: test đơn vị cho từng luật; phát lại trên log dev300 cho đúng các con số trên.

### T1.2 `route()` bỏ entity là đơn vị cơ bản — DONE
- Entity có giá trị (fold) bằng đơn vị cơ bản của mặt hàng duy nhất trong spec không phải điều kiện lọc (spec `q_1885`: `cái` ở cột `Đơn vị tính`,
  role `item`). Đơn vị khác đơn vị cơ bản (vd `thùng`) giữ nguyên: không có hint.
- Bằng chứng: `q_1885` có hint `item_qty(...) -> 89` = gold.

### T1.3 Ép engine hint theo bảng precision — DONE
- Hint của hàm có precision đo được cao (`item_qty` 16/16) đi qua đúng đường `engine_mismatch` của `engine_ref`: báo một lần, Coder giữ số thì được tin.
- Danh sách hàm được ép là dữ liệu cấu hình `DATAMATE_ENFORCE_HINTS` (mặc định `item_qty`), không ép `top_items` / `top_customers` (mẫu 1–2 câu)
  và không bao giờ ép `sales_metrics` toàn shop (64/91).
- Xong khi: test orchestrator (hint lệch → `engine_mismatch` một lần → Coder giữ thì nhận); ghi công tắc vào run meta / `.env.example`.

## T2. Judge bị cắt ở `max_tokens` — DONE
- 8/287 lời gọi judge chạm trần 800 token, 11 lần "unreadable" → bị coi như accept. Thứ tự key `problem, fix` trước `evidence` làm mất phần quan trọng.
- Sửa (bản tối thiểu): `DATAMATE_JUDGE_MAX_TOKENS` (mặc định 1500). Chỉ đổi những lời gọi từng bị cắt; trả lời trung bình ~180 token.
- Đổi thứ tự key trong prompt và parse JSON bị cắt chuyển sang **B4**: cả hai đổi hành vi trên mọi lời gọi judge (Rule 8) và không đo offline được.
- Không nới bộ lọc evidence (xem B1).

## T3. Lexicon bản gọn

Đo trên 4.000 câu: từ vựng train ≠ public (`đh` 0 train / 24 public, `bao nhiu` 185 / 0, `o đờ` 0 / 2); bỏ dấu gây va chạm (`o đờ` = `ở đó` = `o do`);
`đt` ở shop điện thoại là "điện thoại". `norm` chỉ phục vụ bộ đọc regex, **LLM vẫn đọc câu gốc**: lỗi C7 chỉ sửa được bằng chú giải trong prompt.

### T3.1 Khám phá từ lạ — DONE
- `scripts/oov_discovery.py`: token của câu hỏi không có trong lexicon, rulebook, README, tên cột và giá trị dữ liệu của shop, xếp theo tần suất.
- Mỗi câu ghi `unknown_words` vào log Explainer (để log run private cho thấy tiếng lóng mới).

### T3.2 Lexicon duyệt tay — DONE
- `src/datamate/lexicon.py`: bảng `biến thể → từ chuẩn`, áp một lần trong `norm`; mục nào bỏ dấu bị va thì khớp trên chữ có dấu (`o đờ`); không đưa
  mục mơ hồ (`dt`, `đt`). Teencode là từng mục duyệt tay (`j→gi`, `wan→quan`, `cko→cho`, `kua→cua`, `kon→con`, `hok→khong`…), **không** dùng
  luật sinh tự động (đo: `zay→day`, `cko→co`, `zi→di` sai nghĩa).
- Ưu tiên các danh sách miễn trừ (`SKIP`, `BALANCE`, `VERSUS`): thiếu biến thể ở đó thì code ép kỳ sai.
- Xong khi: reading train ≥ 0,959, k 135/135; diff snapshot kỳ train + public đọc từng dòng.

### T3.3 Chú giải vào prompt spec (C7) — DONE
- Mục lexicon có nghĩa (không phải teencode thuần) xuất hiện trong câu → một dòng `WORDS OF THE QUESTION` trong prompt spec, vd `"o đờ" = đơn hàng (order)`.

### T3.4 Telex có kiểm âm tiết — DONE
- Thay guard `len(t) <= 5` của `_telex` bằng kiểm cấu trúc âm tiết tiếng Việt: cắt `s/f/r/x/j` cuối khi phần còn lại là âm tiết hợp lệ và từ gốc thì
  không (`thangs→thang`, `khachs→khach`, `nhieuf→nhieu`; `thanks`, `class`, `plus` giữ nguyên). Bỏ `_THANG` trong `periods.py`.
- Không làm so khớp mờ (đo: 3/8 sửa đúng, 99 cặp từ hợp lệ cách nhau 1 ký tự).

## T4. Từ vựng từ dữ liệu, C6

### T4.1 Đơn vị lấy từ dữ liệu cho luật đếm — DONE
- "bao nhiêu / mấy <đơn vị của shop>" (kg, đĩa, viên, lon, chai, gói, hộp, cây, chiếc…) là câu đếm `number`, đơn vị lấy từ dữ liệu bán hàng của shop.

### T4.2 C6 — DONE
- "mỗi <nhóm> bao nhiêu" với nhóm là kênh / ca / nhóm hàng / người, hoặc "X và Y … (lần lượt | mỗi)" → nhiều số (`one number per group`).
- Xong khi: precision luật trên train ≥ 98%; không đổi đọc sai câu nào khác.

## Backlog

- **B1** Nới bộ lọc evidence của judge: precision revise ~50% (8 revise được nhận: 4 bác nhầm giá trị đúng). Chỉ làm như thí nghiệm, đo bằng
  `p45_report.py` trước khi bật.
- **B2** Nhãn thanh toán lấy từ giá trị cột phương thức thanh toán cho `status._PAID` (hiện là danh sách tay, một phần là đoán).
- **B4** Đổi thứ tự key của judge và đọc JSON bị cắt: một revise bị cắt mà evidence đầy đủ sẽ qua bộ lọc (precision revise ~50%). Đo trước.
- **D1** 15 mục lexicon chỉ thấy trong câu hỏi public (0 câu train): `nv, ncc, ngui, nhứt, mik, mk, wan, jum, jup, trog, cko, kua, kon, kac, tih`; thêm các phương ngữ dò ra nhưng CHƯA thêm: `ni` (= này, 16 câu public), `mô` (= nào, 11), `nhut` (14). Rule 6 cấm dùng test data để train; bảng chính tả viết tay đọc từ câu hỏi public có tính hay không là quyết định của đội. Hiện để trong `lexicon.PUBLIC_SEEN`, tắt mặc định.
- **B5** `find_names` khớp "inside one value" cho cụm ≥ một nửa giá trị (vd `don hang` → giá trị `Trạng thái đơn hàng`). Đã chặn 3 danh từ đo lường bằng `GRAMMAR`; cơ chế chung chưa sửa.
- **B6** `engines/debt.METHOD_COL` không khớp `Thanh toán bằng` (dev_008): kiểm engine công nợ có nhận đúng đơn ghi nợ không.
- **B3** Route `debt_by_customer`, xếp hạng khách theo số đơn, engine `customers(start, end, min_orders)` (E.3 của proposal).

## Nhật ký

- **2026-11-03 T1.1 DONE.** `engines/reference.py`: `resolve_reference(call, tools, answer_type) -> (Reference | None, lý do)`; `build_reference` giữ
  chữ ký cũ. `Reference.key` + `Reference.expr` (`sales_metrics(...)['orders']`), dùng trong `text`, `engine_first` và `reference_note`. Orchestrator
  ghi `engine_call_dropped` (lý do + call) một lần mỗi câu. Replay code thật trên 242 call của dev300: 122 giữ nguyên như run; **59 được sửa, 59/59
  bằng gold**; 4 câu đang sai có ref đúng (`q_2d2b`, `q_e94f` qua `unit` = đơn vị cơ bản; `q_4362` `['orders']` = 66; `q_6292` `['avg_order']`
  = 107.748); 0 câu đúng bị hại. Khác script đo (58) đúng một call: `q_ac89` `unit='thung'` vs `thùng`, code so sau `fold`. Lý do bỏ hay gặp nhất:
  `shift`, `group_by`, `channel`, `customer_group`, `metric='cancelled_orders'`, 4 call sửa được nhưng câu hỏi là phần trăm. 2 test mới.
- **2026-11-03 T1.2 DONE.** `routing._without_base_units`: bỏ entity có giá trị (fold) bằng đơn vị cơ bản của một entity mặt hàng khác, xét theo
  giá trị chứ không theo vị trí; đơn vị khác (vd `bao`) vẫn là điều kiện → không hint. Replay dev300 (hint chỉ khi T1.1 không tạo ref): 1 hint mới
  `q_1885` `item_qty -> 89` = gold (run sai 83); 0 hint bị mất; 10 câu nay có ref T1.1 thay cho hint. Test mở rộng trong `test_engine_hint_...`.
- **2026-11-03 T1.3 DONE.** Issue mới `hint_mismatch` (không dùng lại `engine_mismatch`, vì chữ của nó lấy từ `engine_ref` và rỗng khi chỉ có hint),
  lời nhắc in hint + đơn vị của nó, báo một lần, Coder giữ số thì được tin. `Orchestrator._enforced_hint`: hàm phải nằm trong `enforce_hints`
  (`DATAMATE_ENFORCE_HINTS`, mặc định `item_qty`, ghi vào run meta); với `item_qty`, `spec.unit` phải rỗng hoặc bằng đơn vị cơ bản (fold), vì engine
  đếm theo đơn vị cơ bản. Log `routing` có `enforced`. Replay dev300 (sau T1.1/T1.2): 7 hint `item_qty`, bắn ở **3 câu đang sai** (`q_1885` 89,
  `q_c4eee` 32, `q_daa4` 33, đều = gold), 0 câu đúng. 3 test mới.
- **T1 tổng (replay, chưa chạy model):** 7 câu đang sai nhận đúng một lần gửi lại kèm giá trị engine bằng gold (4 qua ref, 3 qua hint), 0 câu đúng bị
  gửi lại. Trần +7 câu (+0,023 PA); thực tế phụ thuộc Coder có theo không. Cần một lượt dev300 để đo.
- **2026-11-03 T2 DONE.** `Evaluator(max_tokens=...)`, `Settings.judge_max_tokens` (`DATAMATE_JUDGE_MAX_TOKENS=1500`), ghi vào run meta và
  `.env.example`. 1 test. Phần đổi prompt / parser chuyển sang B4.
- **2026-11-03 T3.4 DONE.** `text._telex`: cắt `s/f/r/x/j` cuối của token > 5 ký tự khi phần còn lại là âm tiết hợp lệ (`text.SYLLABLE`) và token thì không.
  Trên 4.000 câu chỉ đổi đúng 4 token norm còn sót: `thangs` ×44, `khachs` ×4, `thoaij`, `nhieuf`. Bỏ `_THANG` trong `periods.py`. Snapshot: train 0 thay đổi.
- **2026-11-03 T3.2 DONE.** `src/datamate/lexicon.py` (bảng `biến thể → chuẩn`, áp trong `text.norm`; biến thể có dấu khớp trước khi bỏ dấu). Không đưa
  `dt/đt`, `hd`, `hong`, `k`, `kh`, `z` (mơ hồ, đo ngữ cảnh từng cái). Đo xong phải sửa thêm 3 chỗ chung:
  (1) `COUNT` bỏ nhánh "nhiều <đối tượng>" trần: trong câu "nào/ai" nó là so sánh nhất (train 11/11 là tên), lexicon làm lộ bug này (`nhiêuf` → `nhieu`);
  (2) `COUNT_SURE` có lại `xe` (lần dọn trước bỏ nhầm cả từ, chỉ cần bỏ guard);
  (3) `GRAMMAR` của `find_names` thêm `doanh thu|don hang|nhan vien`: `dthu hả` → `doanh thu ha` khớp nhầm khách "Thu Hà"; dọn luôn tên rác có từ trước
  (`thu trung`, `vien`, `thu hang`).
  Kết quả (snapshot code cuối so với trước T3): **train reading 0,9590 → 0,9614, 7 sửa đúng / 0 hỏng**, k 135/135, mọi luật sure ≥ 98,9%; public 25 câu đổi,
  tất cả theo hướng mặc định → chắc chắn (15 `đh` thành câu đếm, 10 câu "ai/nào … nhiều nhất" thành tên). Rule 6: 15 mục chỉ có trong public tách ra
  `PUBLIC_SEEN`, tắt mặc định (`DATAMATE_LEXICON_PUBLIC`), xem D1. 6 test mới.
- **2026-11-03 T3.3 DONE.** `QuestionFacts.glossary` + dòng `WORDS OF THE QUESTION` trong prompt spec, chỉ cho mục có `gloss` (`o đờ/oder/order/đh/dh` = đơn hàng,
  `ck`, `tm`); `DATAMATE_GLOSSARY=1`. Log Explainer có `glossary`. Chưa đo trên model (sửa C7 `q_37a4ac` là mục tiêu). 1 test.
- **2026-11-03 T3.1 DONE.** `scripts/oov_discovery.py` (từ không có trong rulebook / README / tên cột / giá trị dữ liệu / lexicon, xếp theo số câu);
  `question.unknown_words` ghi `unknown_words` vào log Explainer (token không phải âm tiết, không phải từ của shop). Lần chạy đầu trên public tìm ra phương ngữ
  `ni`, `mô`, `nhut` ảnh hưởng thẳng tới đọc kỳ / kiểu ("tháng ni", "ngày mô") → chưa thêm, xem D1.
- **2026-11-03 T4.2 DONE.** `question.EACH_OF`: "<nhóm X> và <nhóm Y> … mỗi (bên|cái|loại|kênh|ca|nhóm)" → `list`, rule `one number per group`. Đo trước
  trên train: 7/7 là list (trước đọc number mặc định); "mỗi loại" đứng một mình thì 9/9 là một tổng (giữ number). Snapshot: **train 7 sửa / 0 hỏng →
  reading 0,9638**, k 135/135; public 4 câu đổi sang list (3 câu tuổi nợ "0–7, 8–15 và trên 15 ngày, mỗi nhóm bnhiu", 1 câu "mỗi kênh"). 1 test.
- **2026-11-03 T4.1 DONE.** `question.shop_units(tables)`: giá trị của các cột đơn vị của shop (cùng định nghĩa cột `engines/items.UNIT`), cache theo shop;
  "bao nhiêu / mấy <đơn vị của shop>" (một chữ hỏi) → `number` chắc chắn. Đo trước trên train: 126 câu, tất cả là số. Snapshot: train 143 câu, public 36 câu
  chuyển `default` → `how many`, không câu nào đổi kiểu (reading vẫn 0,9638); luật `how many` 446/446 đúng. Tác dụng chỉ khi spec chọn sai kiểu. 1 test.
- **2026-11-03 Kiểm thêm (T3.2).** `norm` mới cũng đổi việc chọn dòng rulebook cho Coder (`guideline.select`): **159/3.000 câu train** khác khi tắt lexicon,
  gần như toàn bộ là THÊM dòng mà câu viết đủ chữ đã được nhận ("bán được bao nhiu tiền" → dòng "Bán được bao nhiêu tiền" 68 lần, "Doanh thu một đơn" 40,
  "Bán được bao nhiêu <đơn vị>" 28); 8 lần một dòng bị đẩy ra vì giới hạn 4 dòng (`Mặt hàng` 5, `Thuốc kê đơn` 3). Đây là thay đổi prompt Coder, chưa đo
  trên model. `SHIFT_WORDS` (facts.py): 0 thay đổi. A/B: `DATAMATE_LEXICON=0`.

- **2026-11-03 Kiểm cuối.** (1) Cờ lexicon đọc lúc import: `lexicon.py` nay import `datamate.config` trước (nạp `.env`), run meta ghi giá trị đang có
  hiệu lực (`lexicon.ENABLED / GLOSSARY / USE_PUBLIC`), đã thử `DATAMATE_LEXICON=0` → tắt thật. (2) Replay `code_checks` trên 300 spec dev300 với cách đọc
  cuối: chỉ 1 spec bị đổi kiểu (`q_995dd`, list → table = gold, câu run sai), 0 câu đúng bị đụng. 312 test xanh.

## Còn lại / đề xuất bước tiếp
1. **Một lượt dev300** (box vLLM) để đo T1 (trần +7 câu) và T2; A/B `DATAMATE_LEXICON=0` để tách tác động của T3 lên prompt Coder (159 câu đổi dòng rulebook).
2. **D1** (Rule 6): bật hay bỏ `PUBLIC_SEEN`; thêm `ni` / `mô` / `nhut` hay không.
3. B1–B5 theo thứ tự lợi ích: B3 (~5 câu) trước.

### Đợt 2 (2026-11-03, theo yêu cầu: bỏ phân biệt public, làm B1, B2, B4, D1, B5 trước khi chạy lại)
- **D1 DONE.** Một bảng `lexicon.ENTRIES` (49 mục), bỏ `PUBLIC_SEEN` và `DATAMATE_LEXICON_PUBLIC`. Thêm phương ngữ sau khi đọc ngữ cảnh: `ni` (= này,
  17/17), `mô` (= nào, 11/11), `nhut`/`nhứt` (= nhất, 14), `rứa`, `mún`, `bít`; mục tự bảo vệ `mô tô → mô tô` (khớp dài nhất thắng, nên "xe mô tô" giữ
  nguyên). Không thêm `chừ` (periods tự đọc "tới chừ") và `răng` ("kem đánh răng"). Snapshot: train 0 thay đổi (0,9638, k 135/135); public 11 câu "ngày mô /
  hàng mô / khách mô" thành date / string chắc chắn, 15 câu "tháng ni / hôm ni / tuần ni / năm ni" có kỳ đúng, tên rác `rua` (từ "rứa" khớp dịch vụ "rửa xe")
  biến mất.
- **B2 DONE.** `status.payment_labels(tables, roles)`: nhãn của cột phương thức thanh toán của shop (theo tên cột `PAYMENT_COL`, hoặc role README mới
  `payment_method` = "hình thức thanh toán ..."), `rules_for(labels)` thêm chúng vào các từ sau "khách trả". Bỏ các từ đoán (`ck, chuyen, momo, zalo, tm,
  the`): thay bằng `PAYMENT_METHODS` (các phương thức của rulebook) + `ck` / `tm` viết đủ bằng lexicon trước khi phân loại. Loader truyền nhãn vào
  `add_status_std`. Snapshot dữ liệu: **0/30 shop train, 0/14 shop public đổi**. Phát hiện phụ: `engines/debt.METHOD_COL` không khớp cột `Thanh toán bằng`
  của dev_008 (B6). 1 test (cả đường README).
- **B4 DONE.** `JUDGE_SYSTEM` mới đặt `verdict, target, evidence` trước `problem, fix`; `_partial_json` đọc các trường đã đủ của JSON bị cắt (revise vẫn
  cần evidence nguyên vẹn để qua bộ lọc); feedback ghi `[judge reply cut ...]`. Công tắc `DATAMATE_JUDGE_FORMAT=v2` (`v1` = prompt + parser cũ,
  `JUDGE_SYSTEM_V1`). Đổi prompt judge cho mọi câu: chưa đo. 1 test.
- **B1 DONE.** `evaluator.fact_fragments`: evidence tách theo `...` / `vs`, bỏ nhãn (`VERIFIED FACTS:`, `ENGINE VALUE:`, `Code Output:` ...); một mảnh ≥ 2 từ
  nguyên văn trong FACTS / ENGINE VALUE thì revise được giữ (`found_in = facts_fragments`). Mảnh chỉ có trong câu hỏi / code KHÔNG tính (cách đó từng cho lọt
  lời phàn nàn định dạng trên câu đúng). Replay dev300 (chỉ phần ENGINE VALUE, vì chữ FACTS không được log): 2 revise được giữ, cả 2 trên câu sai
  (`q_40de`, `q_c4eee`); nếu có chữ FACTS thì thêm `q_4362` (sai) và `q_b0728` (đúng, `trust_repeat` đỡ). Từ nay log `coder_inputs.facts_text`.
  Công tắc `DATAMATE_JUDGE_EVIDENCE=fragments` (`strict` = như cũ). 1 test.
- **B5 DONE.** `find_names`: khớp "inside one value" cần ≥ 3 từ (`INSIDE_MIN_WORDS`); một giá trị 2 từ viết nguyên vẫn khớp exact. Lý do (đo trên train +
  public): 30 cụm 2 từ khớp bên trong giá trị dài hơn, 29 sai: tên người ("bình quân" → khách "Nghiêm Bình Quân", "trung bình" → "Triệu Trung Bình",
  "là những" → "La Nhung Phong"), nhãn trạng thái ("nhập hàng" → "Đã nhập hàng", "hủy đơn"), sản phẩm "Dây đeo điện thoại" cho kênh "điện thoại"; cụm có
  thể đúng duy nhất là "cát xây" → "Cát xây tô". `name_index` bỏ giá trị trùng tên cột của nó (dòng tiêu đề lọt vào dữ liệu: "Tên khách hàng").
  Snapshot: train reading giữ 0,9638 (0 sửa / 0 hỏng), 54 câu train + 29 câu public bớt tên rác (83 tên); public 2 câu đọc đúng hơn (tên rác "Từ Lan Huy"
  từng nuốt chữ "lần" của "lần lượt"; "có mấy khách trả góp" thành câu đếm). Câu đầu lộ thêm tiếng lóng `xi ô đi` = COD → thêm vào lexicon kèm chú giải.
  1 test.
- **Sửa sau review (đợt 2).** B1: `fact_fragments` bỏ các dòng tiêu đề `#` của FACTS khi so và đòi mảnh phải chứa một con số có trong FACTS (một revise chỉ chép
  tiêu đề "VERIFIED FACTS (counted by ...)" hoặc cụm lặp "the main period" từng qua được bộ lọc). Replay dev300 lại: vẫn đúng 2 revise được giữ (`q_40de`,
  `q_c4eee`, đều trên câu sai), 0 trên câu đúng. B4: chỉ đọc JSON dở khi trả lời thật sự bị cắt (không có `}` sau `{` cuối); JSON hỏng kiểu khác vẫn là
  "unreadable" như cũ. B5: docstring ghi rõ giá phải trả (cụm 2 từ của một sản phẩm, "mì omachi", "cát xây", không còn khớp bằng code; spec call tự tìm).
  Replay issue `entities: the question names ...` trên spec dev300: B5 bỏ 1 issue giả (`q_82ab`, "bình quân" → khách "Trịnh Bình Quân"), không thêm issue nào.

## Chạy lại: cách đọc kết quả
- Nhiễu giữa hai run gần cùng code ~±0,05 PA (BASELINE.md), lớn hơn trần T1 (+0,023): chấm theo từng câu mục tiêu, không theo PA tổng:
  `q_2d2b39ab0b27, q_e94f699997e5, q_4362460c082a, q_629214960e6e` (T1.1), `q_1885ab61433f, q_c4eee7e4ca8d, q_daa477dd5e4a` (T1.2/T1.3),
  `q_995dd5d94d45` (table), `q_40de33fe6cdc` (B1).
- Dấu trong log để đếm: `engine_call_dropped`, `engine_reference` (call đã sửa có `['orders']`...), `hint_mismatch`, `found_in = facts_fragments`,
  `[judge reply cut`, `glossary`, `unknown_words`, `coder_inputs.facts_text`.
- Tắt được bằng cờ (nhánh "gần như cũ"): `DATAMATE_JUDGE_FORMAT=v1 DATAMATE_JUDGE_EVIDENCE=strict DATAMATE_LEXICON=0 DATAMATE_GLOSSARY=0
  DATAMATE_ENFORCE_HINTS= DATAMATE_JUDGE_MAX_TOKENS=800`. KHÔNG có cờ: T1.1 (sửa call), T1.2, `COUNT` / `EACH_OF` / đơn vị, B2, B5, P0/P1 → nhánh cũ thật
  chỉ là run `2b4ca48e19`.
- Phạm vi đổi prompt: prompt judge mới ở mọi câu (B4); 159/3.000 câu train đổi dòng rulebook cho Coder (lexicon); 54 + 29 câu train/public đổi DATA MATCHES
  (B5); dòng chú giải ở các câu có `o đờ / đh / dh / ck / tm / xi ô đi`.
- **B6 DONE.** `engines/debt.METHOD_COL = data.status.PAYMENT_COL` (một định nghĩa, có `Thanh toán bằng`); `build_debt` ưu tiên cột mà README khai báo
  là "hình thức thanh toán" (role `payment_method`). Tiền đề của ticket sai một phần: dev_008 KHÔNG có engine công nợ (README không có câu `**Công nợ**:`,
  nợ ghi trong cột `Note` của đơn), nên việc thiếu tên cột chưa làm sai số nào. Snapshot công nợ 44 shop (10 có engine): **0 thay đổi** (số đơn ghi nợ, tổng
  nợ cuối kỳ, top 3 khách nợ). Test mới: cột `Thanh toán bằng` và cột README khai báo cho cùng tổng nợ 220; không tìm thấy cột thì đơn COD bị tính là nợ
  (300) — lý do phải tìm được cột. Mở **B7**: engine công nợ cho shop ghi nợ trong ghi chú đơn.


---

## P2–P3 (error-analysis-proposal.md), làm 2026-11-03

Thứ tự đã chốt: P2.1 → P2.2 → cổng ref cùng loại → gate §7 → tên người → P2.4 → engine nghiệp vụ. P3.2 (seed, `predict.sh`, Dockerfile) để
riêng. Chưa chạy model: mọi số dưới đây đo offline (snapshot 44 shop trước / sau, replay log dev300, gold train).

**Kiểm đối chứng chung.** Snapshot `BusinessContext.build` của 30 shop train + 14 shop public trước và sau: shop train duy nhất đổi là 5 shop
có engine nợ mới (biz_108/110/119/120/128, train không có câu nợ nào ở các shop này). Không bảng, cột, số đơn, doanh thu nào của train đổi.

### P2.1 Mã đơn từ số thứ tự (harness)
- `data/loader.derive_order_ids`: cột số nguyên mà từ dòng này sang dòng sau chỉ +0 (dòng khác của cùng đơn) hoặc +1 (đơn kế), chỉ được
  đánh lại ở chỗ đổi sheet / ngày bán, ít số hơn số dòng. Bị loại: bộ đếm dòng, cột ngày-trong-tháng, số dòng trong một chứng từ đã có mã
  (biz_119/120 `DanhSachChiTietNhapHang.STT`: bị bắt ở lượt đầu, đã chặn). README: role `order_no` ("số thứ tự đơn") bỏ qua kiểm bước; câu
  "các dòng cùng **STT** là một đơn" chỉ ưu tiên thử trước (câu không nêu bảng: sổ góp của dev_014 cũng có `STT`).
- Thêm cột text `Mã đơn (STT)` = chính số đó (sổ khác tham chiếu "STT 26"); có sẵn mã đơn thì không làm gì. Sau đó `_fill_groups`,
  dòng trùng, `status_std` theo đơn, `sales_orders` chạy như mọi shop.
- "Mỗi dòng một phiếu": không mã đơn + README "mỗi dòng một lượt/phiếu/đơn" hoặc bảng không có cột hàng / số lượng → `SalesView.row_docs`,
  `sales_orders` một dòng mỗi phiếu. README "không ghi tên khách" → mọi đơn là khách lẻ (trước: cột `Làm gì` bị nhận là tên khách, 3.188 "khách").
- Kèm: dòng không còn ô nào sau khi parse là chú thích cuối sheet ("Ghi chú: số liệu ghi tay"), bị bỏ; `IMEI`, `SĐT`, `serial` giữ dạng text.
- Bất biến: dev_011 3.819 đơn = 3.819 ô `Khách trả`; dev_014 6.334 đơn; dev_012 6.060 phiếu (18 dòng chú thích bị bỏ). dev_014 bỏ 175
  dòng trùng trong cùng đơn (40 dòng trùng cả IMEI: lỗi xuất file theo §8).
- Rủi ro đã biết: shop sửa xe giờ trusted, nên `sales_metrics.avg_order` chia cho mọi phiếu, trong khi §6 Sửa xe chỉ chia cho phiếu có thu tiền.
  Public: 3/50 câu dev_012 hỏi "bình quân 1 phiếu (có thu tiền)" có thể nhận ref / hint sai.

### P2.2 Bảng viết tắt (harness)
- `data/aliases.py`: bảng ≤ 2.000 dòng có cột ngắn K và cột tên dài F (unique, tên chứ không phải mã) sao cho cột tên hàng của bảng khác
  được ghi ≥ 80 % bằng K hoặc F (so sau `fold`) và ≥ 5 % chỉ bằng K. Không ghi đè cột gốc: thêm `<cột> (tên đầy đủ)` và các cột loại
  (`Loại`); `item_lines` đếm theo tên đầy đủ; `item_qty("Sạc 25W SS")` = `item_qty("Củ sạc nhanh 25W Samsung")`.
- Biên: dương tính dev_011 0,90 / dev_014 0,80 phần chỉ-viết-tắt; 46 cặp catalog còn lại của 44 shop đều 0,00.

### Cổng "ref cùng loại với đáp án" (thay P2.5, harness)
- `engines/reference.RETURNS`: mỗi hàm khai loại giá trị (money / quantity / orders / count / ratio). `kind_conflict` bỏ ref khi loại khác
  `measure` / `answer_type` của spec (orders ≈ count). Áp cho cả hint. `size_qty` báo lỗi với size không có trong dữ liệu.
- Replay `engine_reference` của 3 run dev300 (20261102-153007, 20261103-050449, 20261103-070503): bỏ 17, cả 17 khác gold
  (`cancel_rate` cho "mấy đơn huỷ", `size_qty('ca chiều')`, `top_customers` cho "nhiều đơn nhất"); giữ 323 ref đúng gold; hint không bỏ cái nào.

### Gate §7 "shop không ghi loại thông tin này" (thay phần từ điển P2.3, harness)
- `agents/coverage.NEEDS`: giới tính, địa chỉ, tồn kho, nhập hàng, giá vốn, nợ, hạn dùng, khách có tên. Mỗi loại = mẫu câu hỏi (trên
  `norm`) + bằng chứng dữ liệu (tên cột tách CamelCase, mô tả cột README, tên bảng, engine; giới tính còn xét nhãn Nam/Nữ). Chạy trước
  Explainer, không cần spec; `DATAMATE_COVERAGE_GATE=fact|on|off` (mặc định fact: chỉ là một dòng VERIFIED FACTS cho Coder; on: trả
  `unanswerable` trước mọi lời gọi model; ghi vào run meta).
- Đo `scripts/coverage_report.py train`: 28 lần bắn, cả 28 gold unanswerable, 0 câu answerable. Hai lỗi bắt được khi đo và đã sửa:
  "dán ống" fold thành "đàn ông"; "Phương" (tên) khớp "phường". Public: 5 lần bắn (khách nữ ×2, khách ở đường ×3); đã sửa 2 lần bắn sai
  thấy được ("nhập thêm máy cũ" là câu bán, "xe khách quen" tính theo biển số).
- Thăm dò thêm (không có trong train): "khách <tên thật của shop> mua bao nhiêu" trên 44 shop, và "khách năm nay / số khách năm 2025".
  Lượt đầu bắn sai hàng trăm lần: họ "Dương" fold thành "duong" (= đường), "năm" fold thành "nam". Đã sửa: "khách đường" đọc trên câu gốc
  có dấu; câu không dấu thì bỏ qua khi chữ sau "khách" là đầu một tên khách có trong dữ liệu; "nam" theo sau bởi năm / nay / ngoái... là
  năm. Sau sửa: 0 lần bắn sai ở địa chỉ / giới tính; train vẫn 28/0, public vẫn 5. Còn lại trong thăm dò là "khách X nợ bao nhiêu" ở shop
  không ghi nợ, đúng §7.
- Đường CLI thật (model không truy cập được): 4 câu train bị chặn ra `unanswerable`, 0 lời gọi model, `check_format.py` /
  `p45_report.py` đọc được log không có `explainer`.
- dev300: bắt được q_95099c4916a8, q_ee9d6f998de7 (khách nữ), q_05a8b30443a1 (shop không ghi khách) — trần, vì đo trên dữ liệu đã nhìn.

### Tên người (P3.1a, harness)
- `text.given_name`: bỏ kính ngữ đứng trước một tên ("A.Đức", "a Đức", "chị Lan"); tên một chữ giữ nguyên ("Anh", "An").
- `data/aliases.find_nicknames`: cột ≤ 60 cách viết, ≥ 90 % dòng là MỘT tên gọi trùng tên gọi của đúng một người trong bảng danh sách
  → thêm `<cột> (họ tên)`. Không áp cho khoá khách (§3). Chỉ dev_012 đổi.

### P2.4 (harness)
- `explainer.recorded_column`: `missing_fields` mà một cột mang đúng tên đó (mọi chữ, theo thứ tự) bị bỏ, có log.
- `facts.py`: kỳ chỉ một phần nằm trong khoảng ngày có dữ liệu → dòng fact "tính trên phần có dữ liệu (§7), không phải unanswerable".

### Engine nghiệp vụ (P3.1b, c — case-specific)
- Nợ ghi trên đơn (`debt.build_snapshot_debt`): cột README mô tả "khách còn nợ của đơn" hoặc ghi chú README nói ghi nợ ở đó
  (`owed_from_note`: "ghi nợ X, đã trả đủ" → 0, "còn thiếu Y" → Y, "đưa trước X, còn lại ghi sổ" → tổng − X...). Chỉ biết số dư ở ngày cuối
  dữ liệu: ngày sớm hơn, `debt_repaid`, `debt_new` báo lỗi rõ. Không kiểm được bằng gold (train không có câu nợ ở các shop này).
- Lô / hạn dùng (`engines/lots.py`): bảng lô (mã lô unique, hạn dùng, số lượng cơ bản), bảng bán theo lô (mã lô + mã đơn), ngày nhập
  (của lô hoặc của phiếu), tiền lô (dòng nhập có mã lô, không thì giá ở bảng lô). Hàm `expiring_lots / expiring_items / expiring_value /
  nearest_expiry`. Train: 10/10 câu hạn dùng của biz_123 / biz_124 khớp gold khi gọi hàm. Dựng ở biz_119/120/123/124, dev_006.
- Hàm không có dữ liệu thì không đưa cho Coder: `top_customers` khi không đơn nào có tên khách (dev_012, biz_112, biz_130),
  `debt_repaid` / `debt_new` khi shop chỉ ghi số dư cuối kỳ.
- Chưa làm: trả góp, bảo hành, hoa hồng thợ, `due_after` cho sổ hạn nợ (dev_004).
