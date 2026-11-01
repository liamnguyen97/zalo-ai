#1 Đáp án giá nhập bình quân sai vì lấy tổng tiền cả phiếu chia cho số lượng một mặt hàng
Câu train `q_00a6e23cbacf` (biz_109, "Tháng 1/2025 Mì Omachi sườn hầm nhập vào giá bình quân một thùng") trả 7.892.815, đáp án đúng 200.556. Model cộng `Cần trả NCC` (tổng tiền cả phiếu nhập, gồm mọi mặt hàng trên phiếu) rồi chia cho số lượng của riêng Omachi. Thông tin cần thiết đều có trong prompt (README, rulebook, thẻ schema) nhưng không nằm cạnh bảng mà nó thuộc về: không chỗ nào nói bảng dòng nhập là con của bảng đầu phiếu, cột nào chỉ có ở đầu phiếu, và `Cần trả NCC` là tổng cả phiếu. Công thức sai được viết ngay trong `assumptions` của spec rồi Plan và Coder làm theo; judge vẫn chấp nhận.
Đã làm: module mới `data/structure.py` nhận diện từ dữ liệu (không dựa tên cột) bảng cha/con, cột tổng cả phiếu (bằng tổng các dòng con ở ≥ 95% số phiếu), cột cấp phiếu của bảng phẳng, câu README định nghĩa cột. Thẻ schema ghi `PARENT of` / `CHILD of ...; only the parent has ...`, `DOCUMENT TOTAL = sum ...: never divide it by one item's quantity`, `README: "..."`. Trên 30 business train bắt được 15 business có cha/con, 30 cột tổng ở 11 business (cả 7 business có phiếu nhập), thẻ dài thêm 16%. Công tắc A/B `DATAMATE_STRUCTURE_CARDS=0`. CHƯA đo hiệu quả trên model: tập `splits/purchase12` (12 câu nhập hàng) đã chuẩn bị, chưa chạy. Chưa kiểm tra trên 14 business public test (không có dữ liệu trên máy này).
---------------------

#2 Explainer mất 2 lượt vì dùng tên bảng trong README, không phải tên thật
README gọi bảng là `DanhSachNhapHang` nhưng loader đặt key là `DuLieuBanHang__DanhSachNhapHang` (tên file + tên sheet). Turn 0–1 model gọi lookup bằng tên README và nhận `no table`. Lệch tên này có ở 4/30 business (biz_109, 116, 121, 123). `find_values` cũng chỉ trả "no value" khi hỏi tên sản phẩm trong bảng dòng nhập (bảng đó chỉ chứa mã hàng).
Đã làm: hàm `resolve_table` khớp tên README về key thật khi chỉ có một bảng trùng; dùng ở lookup của Explainer, `repair_spec`, Column Selector và guard sandbox. Thẻ ghi `README name: ...`. `find_values` tìm thêm ở các bảng khác khi bảng được hỏi không có giá trị và báo rõ. Có test.
---------------------

#3 Cột đặt nhầm bảng khiến spec kẹt ở `unchecked` và không được kiểm độc lập
Spec turn 3 đặt `Thời gian` (ngày phiếu) vào bảng dòng nhập, nơi cột này không tồn tại. `repair_spec` chỉ bỏ cột và báo "has no columns", model không biết cột nằm ở đâu, turn cuối lại chỉ xin lookup nên spec kết thúc `unchecked`. Trong code, khi còn lỗi thì clarity check không chạy (`if errs: ... elif ... check`), nên spec sai công thức chưa từng được kiểm.
Đã sửa: (1) `repair_spec` tự chuyển cột sang bảng có liên kết cha/con và ghi chú kèm cách join; phát lại spec turn 3 của câu Omachi cho 0 lỗi. (2) Vòng Explainer (`explainer.py`) chạy blind_type và clarity check cả khi spec chỉ còn lỗi "bỏ bảng/cột không tồn tại" (lỗi khác như kỳ sai, entity không có trong dữ liệu vẫn không tốn lượt check); nếu check sạch thì nhận spec luôn, nếu có vấn đề thì gửi cả lỗi lẫn vấn đề về model. (3) Nếu hết ngân sách mà chưa có spec sạch, thêm đúng một call `FINAL TURN` đóng lookup, bắt trả spec đầy đủ (`_forced_spec`); spec đó thay spec cũ khi không nhiều lỗi hơn. Có 4 test mới trong `tests/test_pipeline.py`. Chưa đo trên model: xem kết quả chạy 10 câu bên dưới.
---------------------

#4 Guard sửa tên bảng sai hoa/thường nhưng sửa không bao giờ tới sandbox
`_check_table_names` sửa tên sai chữ hoa/dấu trên cây AST, nhưng code chạy trong sandbox vẫn là bản gốc (chỉ unparse khi có thay đổi khác). Lỗi tồn tại từ trước, bản sửa chưa bao giờ có tác dụng.
Đã làm: hàm trả về số tên đã sửa và kích hoạt unparse lại. Có test (cả trường hợp chỉ sai hoa/thường).
---------------------

#5 Guard chặn nhầm merge bảng nhập hàng với đầu phiếu
`_check_item_attribute_merge` thấy `.merge()` cùng chuỗi `'Tên hàng'`, `'ĐVT'` nên tưởng là merge bảng bán hàng với bảng sản phẩm. Ở câu Omachi đó là merge dòng nhập với đầu phiếu nhập, block 1 bị từ chối, mất 77 giây; block 2 gửi lại y hệt thì được chạy.
Đã sửa: guard biết bảng dòng bán hàng mà `item_lines` được dựng từ đó (`ItemView.table` → `item_lines.attrs["source_table"]` → `GuardedExecutor.sales_table`), theo dõi nguồn gốc hai vế của `.merge()` / `pd.merge()` qua các phép gán (`d = tables['x']`, `d2 = d[...]`) và chỉ chặn khi một vế dựng từ `item_lines`, `sales_orders` hoặc bảng dòng bán hàng. Chưa biết bảng bán hàng thì giữ luật cũ. Phát lại block 1 của câu Omachi: không còn bị chặn. Có test mới (`tests/test_guards.py`).
---------------------

#6 Chạy local trên Ollama rất chậm (~6,6 phút/câu)
Câu đầu tiên mất 394 giây: 12 call model, 102.819 token vào (48% dùng lại cache), 4.153 token ra; Explainer chiếm 48% thời gian (5 call), Coder 33%. Nguyên nhân là cấu hình thiết kế cho RTX 5090 + vLLM chạy song song; M3 Pro đọc prompt ~300 token/giây, sinh ~20 token/giây, một request mỗi lần. Run 20 câu đã dừng ở 12/20 câu xong (log còn trong `result/runs/20261101-164639_ollama-train20/`).
Đã làm: thêm backend `ollama` (`config.py`, `llm.py`, `cli.py`; thinking tắt bằng `reasoning_effort:"none"`, vì `think:false` không có tác dụng), model dẫn xuất `qwen3.5-datamate` (`scripts/ollama/Modelfile`: ctx 65536, `presence_penalty` 0), script `scripts/ollama_run.sh` với `SPLIT=train20|purchase12`, tắt thinking của Plan để nhanh hơn (khác cấu hình `BASELINE.md`, nên kết quả local không so trực tiếp được). Hướng giảm call (bỏ `describe` thừa, tra giá trị bằng code trước turn 0) chưa làm.
---------------------
