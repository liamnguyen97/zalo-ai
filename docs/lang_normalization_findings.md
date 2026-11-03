# Chuẩn hoá viết tắt / teencode và rewrite tiếng Anh ở pha Explainer (2026-11-03)

Hai câu hỏi: (1) có nên dùng từ điển chuẩn hoá câu hỏi viết tắt / teencode thành tiếng Việt chuẩn trước khi xử lý; (2) có nên dịch câu
hỏi sang tiếng Anh khi schema, rulebook, dữ liệu đều tiếng Việt. Mọi số dưới đây **đo offline** trên log có sẵn (không gọi model):
run `20261102-153007_dev300` (fast P2, PA 0,760) và log v3.5 (`result/runs/logs`, PA 0,840), cùng 300 câu dev300 (đã tune trên đó),
cộng 3.000 câu train và 1.000 câu public (chỉ đếm tần suất, không có gold public trên máy). Đề xuất 1–5 đã làm cùng ngày (mục
**Đã làm** cuối file); harness thực nghiệm `scripts/lang_ab.py` và `scripts/lang_offline/` đã xoá, chưa từng chạy với model.

## Hệ thống đang làm gì (đọc code)

- "Dịch" thực chất là field `rewritten_question` của spec (tiếng Anh). Tên hàng/khách, cột, đơn vị, nhãn trạng thái giữ nguyên văn
  tiếng Việt. Coder nhận **cả** `ORIGINAL QUESTION` (tiếng Việt) lẫn `Compute: <rewrite tiếng Anh>`; judge cũng thấy cả hai.
- Đã có từ điển: `lexicon.py` (bảng biến thể → từ chuẩn; lúc đo, `PUBLIC_SEEN` tắt mặc định vì Rule 6) và `_telex` trong `text.py`. Nhưng chúng
  chỉ phục vụ các bộ đọc regex (`norm`); **LLM vẫn đọc câu gốc**, chỉ thêm dòng chú giải cho 4 mục (`o đờ`, `đh`, `ck`, `tm`).

## Câu 1: chuẩn hoá bằng từ điển

| Thí nghiệm (offline) | Kết quả |
|---|---|
| E1 tần suất | Câu "nhiễu": 54,2% train, 57,2% public. Không dấu hoàn toàn 25,8% / 26,2%; có viết tắt của lexicon 42,5% / 37,9%; lỗi gõ telex 10,0% / 2,2% |
| E1 lệch từ vựng | 13 biến thể chỉ có ở public, 0 lần ở train: `mik` 36, `dh` 28, `đh` 24, `nhứt` 10, `wan` 7, `trog`/`cko`/`kua`/`kon`/`jum` 6... Từ điển dựng từ train không phủ được tập test mới |
| E2 giữa các câu | PA(nhiễu) − PA(sạch): run fast +0,036 (CI 95% [−0,11; +0,18]); v3.5 −0,038 ([−0,16; +0,08]). Hai run ngược dấu, đều chứa 0 |
| E4 so cặp trong cùng template | 42 template có cả bản sạch và bản nhiễu: bản nhiễu tệ hơn 5, tốt hơn 5, bằng 32 (fast); 3 / 5 / 34 (v3.5) |
| E2b kiểu đáp án của spec | Đúng gold 98,0% (câu nhiễu) vs 99,3% (câu sạch) |
| Đọc tay 39 câu sai/bỏ trống (fast) | 1 câu sai rõ do viết tắt: `q_f19c4e273491` "quý 4/2025 cuwar hàng có bn order ddos" → "which product had the most orders". 2 câu có phần do cách viết: `q_bf22965b9ea7` (không dấu, "gan het han" hiểu thành "SL còn ít nhất"), `q_995dd5d94d45` ("moi nguoi con no bnhiu" ra list thay vì table; code đọc kiểu hiện đã sửa) |

Lỗi tìm thấy: `norm` chạy `_telex` **trước** lexicon, nên `order` → `orde`, `oder` → `ode`; hai mục `order`/`oder` của lexicon không
bao giờ khớp. Vì vậy `q_f19c` vẫn đọc kiểu "default" thay vì "how many". Ngoài ra `_telex` biến tên thương hiệu thành chữ khác
(`Tiger` → `tige`, `Gas` → `ga`, `Max` → `ma`, `Dulux` → `dulu`): vô hại khi so tên (cả hai phía cùng qua `norm`) nhưng không được đưa
cho LLM đọc. Đã sửa (đề xuất 3).

Kết luận: model 9B đã đọc viết tắt / không dấu tốt; trên dev300 không đo được thiệt hại do cách viết (trần lợi ích ≤ 1–3 câu, số đo trên
dữ liệu đã tune). **Không thay câu hỏi bằng câu đã chuẩn hoá**: `_person_in_question`, `keep_anchored`, quote của judge đều so chuỗi con
với câu gốc. Giá trị thật của chuẩn hoá là độ bền với từ lạ ở private, và từ điển đóng có trần độ phủ (lệch train → public ở trên).

## Câu 2: rewrite tiếng Anh

| Thí nghiệm (offline) | Kết quả |
|---|---|
| E3 tên riêng | 79/79 tên thật (≥ 2 từ) giữ nguyên văn trong rewrite tiếng Anh; 2 "tên" còn lại là khớp nhầm ("gần hết hạn", "bình quân" ↔ khách "Trịnh Bình Quân") |
| E3 chuỗi Coder dùng để lọc | Không ca nào Coder lọc bằng chữ tiếng Anh dịch từ giá trị dữ liệu; chữ tiếng Anh chỉ xuất hiện trong code dò cột (gender, staff...) ở câu gold unanswerable |
| Phân loại 39 câu sai/bỏ trống | 4 rewrite lệch nghĩa (`q_83273acf61f7` tự thêm "prescription" → 0; `q_bf22`; `q_995d`; `q_f19c`); 2 spec hỏng (rewrite = câu gốc); 2 sai kỳ/entity; **~31 spec đúng nghĩa, sai ở Coder/dữ liệu** |
| Lịch sử "tiếng Anh tăng điểm" | v1 (prompt Việt) 0,517 → v2 (prompt Anh) 0,707 đổi cùng lúc rất nhiều sửa lỗi; BASELINE.md tự ghi phần chênh công bằng +0,047, nằm trong nhiễu ±0,05 (≈30 câu lật giữa hai run gần như cùng code) |
| Coupling ẩn | `engines/routing.py` đọc rewrite tiếng Anh bằng regex tiếng Anh: `NOT_QUANTITY` (revenue, price, stock...) chặn hint `item_qty`, chữ "customer" mở hint `top_customers` (12 spec dev300 có hint này nhờ chữ "customer"). `coder.PAIR_FORM` đọc `answer_form` tiếng Anh. Rewrite tiếng Việt sẽ âm thầm tắt các cổng này |

Kết luận: nỗi lo "dịch làm agent rối" không thấy trong log: tên giữ nguyên văn, Coder lọc bằng giá trị tiếng Việt, và lỗi chính không
nằm ở ngôn ngữ (chỉ ~1–2/300 câu có thể quy cho bản dịch, `q_8327` rõ nhất). Phần "tiếng Anh tăng điểm" chưa được chứng minh tách
riêng, và một phần đến từ việc code phía sau đã được viết cho rewrite tiếng Anh.

## Đề xuất

| # | Đề xuất | Loại | Đo bằng |
|---|---|---|---|
| 1 | Giữ thiết kế song ngữ hiện tại (chỉ dẫn tiếng Anh, câu gốc tiếng Việt luôn đi kèm, tên/cột nguyên văn). Không chuyển rewrite sang tiếng Việt trước khi A/B | harness | A/B rewrite Việt / Anh (harness đã xoá; cần port 2 cổng routing trước, nay đã làm ở ý 2) |
| 2 | Code phía sau đọc field có cấu trúc (`measure`, `entities[].role`, `answer_type`) thay vì regex tiếng Anh trên `rewritten_question` (`routing.NOT_QUANTITY`, `"customer" in rewrite`). Ngoài coupling ngôn ngữ, cổng chữ "customer" còn bắn nhầm: `q_4333ad0589a1` (3 khách **nợ** nhiều nhất) được đưa hint `top_customers` (xếp theo **doanh thu**) ở cả 2 lượt (đo trong log dev300) | harness | replay routing trên spec dev300: hint đúng giữ nguyên, hint nợ ↔ doanh thu biến mất |
| 3 | Sửa thứ tự trong `norm`: lexicon trước `_telex` (hoặc `_telex` bỏ qua token là biến thể của lexicon) | harness (sửa bug) | `tests/test_question_reading.py` + `q_f19c` đọc "how many" |
| 4 | Thêm MỘT dòng "câu đã viết đầy đủ" cạnh câu gốc trong prompt spec, sinh bằng cơ chế chung: lexicon (chỉ train) + giải mã telex (phương pháp gõ, không phải danh sách từ), chỉ nhận khi ra một âm tiết tiếng Việt hợp lệ, bỏ qua từ có trong dữ liệu shop và chữ viết hoa giữa câu. Đo offline: đổi 41,5% câu train / 35,7% public; đọc tay danh sách sửa: ~98% lần giải telex đúng (sai còn lại: tên thương hiệu như `Power`, `Max`, đã chặn bằng luật chữ hoa) | harness | `DATAMATE_WRITTEN_OUT` bật / tắt trên dev300 |
| 5 | Spec ghi `conditions: [{"quote", "meaning"}]`, quote phải là chuỗi con của câu gốc (cơ chế quote đã dùng cho judge/checker); điều kiện không có quote bị bỏ. Chặn lỗi thêm điều kiện kiểu "prescription", "cancelled **and returned**" | harness | số rewrite lệch nghĩa trên câu bất đồng giữa arm |
| 6 | T3.1: log từ lạ mỗi câu (`unknown_words`) để run private cho thấy tiếng lóng mới; không mở `PUBLIC_SEEN` vì Rule 6 | harness (công cụ) | log run public / private |

Không đề xuất: khôi phục dấu bằng từ điển cho câu không dấu hoàn toàn ("ban" = bán/bàn/bạn); câu không dấu đã không kém câu có dấu (E2).

## Đã làm (2026-11-03)

| # | Thay đổi | Kiểm tra (offline, không gọi model) |
|---|---|---|
| 1 | Giữ thiết kế song ngữ: không đổi code | — |
| 2 | `engines/routing.py` bỏ regex tiếng Anh trên rewrite (`NOT_QUANTITY`, chữ "customer"). Hint chỉ đưa khi: (a) `engine_call` của spec không chọn hàm khác (kể cả khi hàm đó không chạy được ở shop); (b) spec không trích riêng section nợ / kho của rulebook (`Spec.rule_sections`, code đọc từ `rule_refs`; trích kèm §1/§3 thì không chặn, vì model hay trích §5 "Nhập trừ bán" chỉ để đổi đơn vị); (c) câu gốc (đã `norm`, bỏ tên entity) không nói tồn / nhập / nợ / trung bình (`NOT_SOLD`), không nói tiền với `item_qty` (`MONEY`), và nói "khách" / "ai mua" với `top_customers` (`CUSTOMERS`) | Replay `route()` trên spec 2 run dev300: mọi hint giữ nguyên (38/38 mỗi run), chỉ đổi câu nợ: `q_4333ad0589a1` (cả 2 run) và `q_2d2d91435d5d` (tuổi nợ) không còn hint `top_customers`. Train, 294 câu "bao nhiêu <đơn vị>": không chặn câu số lượng bán nào; chặn tồn kho và 4 câu "nhập trừ bán". Xếp hạng theo tiền: 87 câu khách mở hint, nhân viên / nhà cung cấp / ngày thì không |
| 3 | `text.norm`: lexicon chạy trước `_telex` và một lần sau | Snapshot 4.000 câu (norm, tên, kỳ, kiểu, dòng rulebook, từ lạ): 6 câu đổi, cả 6 đúng hơn (`order` → "don hang": `q_f19c` đọc "how many" chắc chắn; `bữa ni` → "hom nay"). Đọc kiểu train giữ 0,9638 |
| 4 | Dòng `QUESTION WITH ITS SHORTHAND WRITTEN OUT` trong prompt spec (`agents/question.written_out`, `text.telex_decode`); câu gốc vẫn là chuẩn. Biến thể không dấu so trên chữ đã bỏ dấu như lexicon ("đc" → "được"), nhưng dấu thanh phải khớp chữ chuẩn ("kó" = có, không viết thành "không"); từ của dữ liệu shop so ở dạng `fold_text` (không qua `norm`) | Đổi 44,8% câu train / 40,6% public. So với bộ mở rộng đã đọc tay: 264 câu khác, đọc hết danh sách chữ đổi: chữ có dấu nay được viết đủ (`đc` → được 149, `nhìu` → nhiều 30, `ni` → này 17, `kòn`/`kủa`/`jùm`/`tíh`/`wán`...), chữ hoa đầu câu (`Dthu` → Doanh thu), tên hàng được giữ (`Sino SP D20`, `Max`). `bữa nay` giữ nguyên (dấu ngã không có trong "hôm nay") |
| 5 | Field `conditions: [{"quote", "meaning"}]` trong spec; quote phải là chữ của câu gốc (so `fold_text`, hoặc `norm` hai phía). Điều kiện không có quote bị bỏ và đưa vào lần gọi sửa sẵn có; điều kiện hợp lệ hiện cho Coder ("these are ALL the filters"). Log `fix_for_conditions_only` khi chỉ nó gây lần gọi sửa | Test với spec giả kiểu `q_8327` ("kê đơn" không có trong câu → 1 lần sửa, điều kiện bị bỏ). Chưa đo được trên model |

Công tắc (ghi vào `settings` của run): `DATAMATE_WRITTEN_OUT` (ý 4), `DATAMATE_SPEC_CONDITIONS` (ý 5), mặc định bật. Ý 2, 3 là sửa lỗi,
không có công tắc. Test: 327 pass (thêm 10).

Còn mở:
- **Chưa chạy model.** Ý 4, 5 cần một lượt dev300 bật / tắt công tắc. Đo: số rewrite lệch nghĩa, `fix_for_conditions_only` (chi phí thêm),
  kiểu đáp án của model trước khi code sửa. Với ±0,05 nhiễu giữa hai run, PA end-to-end trên 300 câu không đủ để kết luận.
- **Rule 6**: từ đợt 2 (`tickets.md` D1) bảng `lexicon.ENTRIES` đã gộp các biến thể chỉ thấy ở public (`mik`, `kua`, `cko`, `trog`,
  `wan`, `kon`, `nhứt`...) theo quyết định của nhóm. Vì vậy các con số "từ điển dựng từ train" ở trên (E1, độ phủ train → public) mô tả
  bảng lúc đo, và một tập câu nhiễu "heldout" dựng từ các biến thể này không còn đo được khả năng tự đọc của model.
- **`kó` → "khong" trong `norm`** (chưa sửa: bảng lexicon thuộc phần việc khác): biến thể không dấu `ko` (= không) khớp chữ có dấu
  `kó` (= có) sau khi bỏ dấu, đảo nghĩa câu. Gặp ở 2 câu public (`q_338d0179f66d` "shop mik kó mí ngừi...", `q_5fcac0749f24`
  "kó bnhiu món"). Cách sửa chung: biến thể không dấu chỉ khớp chữ có dấu thanh nằm trong dấu thanh của chữ chuẩn, như `written_out` đã làm.
