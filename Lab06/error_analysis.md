# Error analysis (LAB 06, mục 27)

> Mọi số liệu lấy từ `self_attention.ipynb` (mục C) và lưu ở `error_analysis_evidence.json`; model: **Self-attention seed 42** (B8, accuracy test 0.8066). Nhãn nhóm lỗi và "token quan trọng" là giả thuyết của mình, được kiểm tra bằng xóa token.

## Phương pháp

* 6 ca chọn **thủ công** từ tập test (n = 10 000) sau khi xem danh sách ứng viên: 3 sai, 3 đúng. Không ngẫu nhiên nên **không đại diện** cho phân phối lỗi.
* Model chỉ thấy **100 token đầu** của mỗi review (cùng cấu hình LAB 05). Cột `token thấy/tổng` cho biết bị cắt bao nhiêu.
* "Top-3 attention" = 3 token nhận nhiều attention nhất, trung bình theo cột của ma trận attention `T × T` (cách tóm tắt, không phải cách duy nhất).
* Kiểm chứng giả thuyết bằng **xóa token** rồi đo lại `p(positive)` (và log-odds, vì p gần 0/1 nên p khó thấy thay đổi): xóa top-1, xóa top-3, xóa nhóm token "người đọc cho là quan trọng", và **đối chứng** (xóa 3 token ngẫu nhiên, 100 lần). Với review dài hơn 100 token, thêm thử "chỉ đưa 100 token cuối".
* Hạn chế: xóa token làm đổi vị trí các token sau nó (có position embedding) và đổi phân phối đầu vào, nên đây là bằng chứng gợi ý, không phải bằng chứng nhân quả.

## Bảng tổng hợp

| # | sample_id | Nhãn thật | Dự đoán | p(pos) | log-odds | token thấy/tổng | Nhóm lỗi | Top-3 attention |
|---|---|---|---|---|---|---|---|---|
| 1 | provided_csv:20897 | neg | **pos** (sai) | 0.9996 | +7.84 | 42/42 | negation / ambiguous wording | pleasantly (0.075), cameos (0.049), the (0.046) |
| 2 | provided_csv:112 | neg | **pos** (sai) | 0.9891 | +4.51 | 41/41 | sarcasm (kèm noise: câu viết lủng củng) | terrific (0.098), fun (0.084), better (0.055) |
| 3 | provided_csv:25358 | neg | **pos** (sai) | 0.9909 | +4.69 | 100/742 | truncation / long-range dependency | great (0.033), numerous (0.026), in (0.025) |
| 4 | provided_csv:35024 | neg | neg (đúng) | 0.0085 | −4.76 | 51/51 | — | cheap (0.096), painful (0.047), very (0.042) |
| 5 | provided_csv:6028 | pos | pos (đúng) | 0.9996 | +7.93 | 42/42 | — | composition (0.090), our (0.047), great (0.046) |
| 6 | provided_csv:3286 | pos | pos (đúng) | 0.7203 | +0.95 | 51/51 | — (đúng nhưng kém chắc chắn, có "not much more") | much (0.052), all (0.049), entertained (0.039) |

### Kết quả kiểm tra bằng xóa token (log-odds của p(positive); p trong ngoặc)

| # | gốc | xóa top-1 | xóa top-3 | xóa nhóm giả thuyết | đối chứng: Δp khi xóa 3 token ngẫu nhiên | chỉ 100 token cuối |
|---|---|---|---|---|---|---|
| 1 | +7.84 | `pleasantly`: +5.77 (0.997) | +4.28 (0.986) | `not, worth`: +5.94 (0.997); `pleasantly, surprised`: +4.85 (0.992) | −0.002 ± 0.003 | — |
| 2 | +4.51 | `terrific`: +0.87 (0.706) | +0.34 (0.584) | `terrific`: +0.87 (0.706); `boring, pity`: +3.53 (0.971) | −0.083 ± 0.100 | — |
| 3 | +4.69 | `great`: +4.02 (0.982) | +4.39 (0.988) | `great, popular`: +4.17 (0.985) | −0.000 ± 0.004 | **−0.38 (0.407)** |
| 4 | −4.76 | `cheap`: −3.19 (0.040) | −2.67 (0.065) | `cheap, painful`: −2.20 (0.100); `avoid`: −4.71 (0.009) | +0.002 ± 0.012 | — |
| 5 | +7.93 | `composition`: +7.29 (0.999) | +5.90 (0.997) | `great, fantastic, superb`: +5.03 (0.994) | −0.000 ± 0.001 | — |
| 6 | +0.95 | `much`: +1.17 (0.763) | +1.98 (0.879) | `entertained`: +1.59 (0.830); `not, much, more`: +1.44 (0.808) | **+0.104 ± 0.051** | — |

### Thống kê mô tả trên toàn tập test (không nhân quả)

| nhóm | n | accuracy Self-attention | accuracy LSTM (cùng nhóm, seed 42) |
|---|---|---|---|
| ≤ 100 token (không bị cắt) | 1141 | 0.8791 | 0.8344 |
| > 100 token (bị cắt) | 8859 | 0.7973 | 0.7485 |
| > 300 token | 2340 | 0.7385 | 0.6880 |
| có token phủ định trong 100 token đầu | 7158 | 0.8094 | 0.7557 |
| không có token phủ định | 2842 | 0.7994 | 0.7650 |

## Từng ca

### Ca 1 — `provided_csv:20897` (sai: neg → pos, p = 0.9996)
Đoạn đầu: *"definitely not worth the rental but if you catch it on cable you'll be pleasantly surprised by the cameos …"* (review chỉ 42 token, không bị cắt).

* **Nhóm lỗi:** negation / ambiguous wording (review nêu kết luận âm "not worth the rental" rồi một ngoại lệ dương "pleasantly surprised on cable").
* **Token người đọc cho là quan trọng (giả thuyết):** `not worth the rental` (phán quyết âm), `pleasantly surprised` (ngoại lệ).
* **Attention có tập trung vào các token đó không?** Top-3 là `pleasantly`, `cameos`, `the`: có chạm `pleasantly`, không chạm `not` / `worth`. Mức attention thấp (≤ 0.075 so với 1/42 ≈ 0.024 nếu đều), nên không nhọn.
* **Giả thuyết (không phải kết luận):** model dựa nhiều vào cue dương và không xử lý "not worth" như phủ định.
* **Đối chiếu với hành vi model:** xóa `pleasantly, surprised` giảm log-odds từ +7.84 xuống +4.85 (ảnh hưởng dương đúng như giả thuyết), nhưng p vẫn 0.992, tức dự đoán không đổi. Xóa `not, worth` cũng *giảm* log-odds (+7.84 → +5.94), tức hai token này (gộp lại) đang đẩy về phía **dương**, không phải âm; số liệu không tách được riêng `not` khỏi `worth`. Đối chứng chỉ đo bằng Δp (−0.002 ± 0.003) trong khi p bão hòa gần 1, nên không so trực tiếp được với thay đổi log-odds; ở đây chưa có đối chứng theo log-odds.
* **Nhận định:** Mình xếp ca này vào negation / ambiguous wording vì review vừa kết luận âm ("not worth the rental") vừa có ngoại lệ dương ("pleasantly surprised" khi xem trên cable). Attention chạm `pleasantly` nhưng không chạm `not`/`worth`, và dự đoán dương rất chắc (p = 0.9996). Phép xóa chỉ ủng hộ một phần: bỏ `pleasantly, surprised` làm log-odds giảm +7.84 → +4.85 nhưng p vẫn 0.992, và bỏ `not, worth` cũng làm log-odds giảm (+7.84 → +5.94), tức hai token này gộp lại đang đẩy về phía dương chứ không phải âm. Vì vậy mình chỉ kết luận model dựa vào cue dương và chưa thấy bằng chứng nó xử lý "not worth" như phủ định; chưa chứng minh được nguyên nhân.

### Ca 2 — `provided_csv:112` (sai: neg → pos, p = 0.9891)
Đoạn đầu: *"it's terrific when a funny movie doesn't make smile you what a pity this film is very boring and so long …"* (41 token).

* **Nhóm lỗi:** sarcasm (mở đầu bằng "terrific", "funny" theo kiểu mỉa) + noise (câu sai ngữ pháp, `br br`).
* **Token quan trọng theo người đọc (giả thuyết):** `pity`, `boring`, `no fun`, `painfull` (chê); `terrific`, `funny` là cue gây nhầm.
* **Attention:** top-3 là `terrific`, `fun` (trong cụm "no fun"), `better`; không có `boring`/`pity`.
* **Giả thuyết:** model bị kéo bởi cue dương ở đầu câu và không nhận ra giọng mỉa.
* **Đối chiếu:** xóa `terrific` kéo log-odds từ +4.51 xuống +0.87 (p 0.989 → 0.706); xóa top-3 còn +0.34 (p 0.584). Đối chứng chỉ −0.083 ± 0.100, nên hiệu ứng của `terrific` lớn hơn nhiều so với xóa ngẫu nhiên: **ủng hộ** giả thuyết rằng `terrific` là bằng chứng dương quan trọng. Ngược lại, xóa `boring, pity` làm log-odds *giảm* (+4.51 → +3.53) thay vì tăng như ta kỳ vọng nếu chúng là cue âm. Điều này **không ủng hộ** việc model coi `boring`/`pity` là cue âm rõ ràng; có thể do phép xóa làm lệch vị trí và pooling (hạn chế đã nêu), nên không kết luận thêm.
* **Nhận định:** Mình xếp ca này vào sarcasm (kèm noise do câu viết lủng củng). Attention tập trung vào `terrific`, `fun`, `better`, không có `boring`/`pity`. Xóa `terrific` làm log-odds từ +4.51 xuống +0.87 (p 0.989 → 0.706), lớn hơn nhiều so với đối chứng (−0.083 ± 0.100), nên giả thuyết "model bị cue dương `terrific` kéo" được ủng hộ. Ngược lại, xóa `boring, pity` làm log-odds giảm (+4.51 → +3.53) chứ không tăng, nên không có bằng chứng model coi chúng là cue âm rõ ràng; phép xóa còn làm lệch vị trí và pooling nên không kết luận thêm.

### Ca 3 — `provided_csv:25358` (sai: neg → pos, p = 0.9909)
Review dài 742 token, model chỉ thấy 100 token đầu, mở đầu: *"the author of numerous novels plays and short stories w somerset maugham … was considered among the world's great authors …"* (giới thiệu, chưa có đánh giá).

* **Nhóm lỗi:** truncation / long-range dependency.
* **Token quan trọng theo người đọc (giả thuyết):** phần đánh giá (không có trong 100 token đầu), vì phần thấy được chỉ là tiểu sử ca ngợi tác giả.
* **Attention:** top-3 `great`, `numerous`, `in`, rất thấp (≤ 0.033 so với đều 0.01), tức attention rất phẳng.
* **Giả thuyết:** nhãn âm nằm ngoài cửa sổ 100 token nên model chỉ có từ khen của đoạn giới thiệu.
* **Đối chiếu:** xóa top-3 hoặc `great, popular` gần như không đổi (log-odds +4.69 → +4.39 / +4.17). Khi đưa **100 token cuối** thay vì 100 token đầu, p(pos) = 0.407 (log-odds −0.38), nghĩa là dự đoán đổi sang phía âm (đúng nhãn). **Ủng hộ** giả thuyết truncation cho riêng ca này (một ca; p chỉ 0.407, chưa chắc chắn). Thống kê mô tả: accuracy 0.8791 (≤ 100 token, n = 1141) so với 0.7973 (> 100 token, n = 8859); nhóm > 300 token là 0.7385. Chênh lệch này không tách được độ dài khỏi các yếu tố khác (phong cách review, tỉ lệ nhãn…).
* **Nhận định:** Mình xếp ca này vào truncation / long-range dependency: review dài 742 token, model chỉ thấy 100 token đầu, mà đoạn này là tiểu sử ca ngợi tác giả nên không có đánh giá. Attention rất phẳng (top-1 chỉ 0.033, đều là 0.01). Xóa top-3 hay `great, popular` gần như không đổi (log-odds +4.69 → +4.39 / +4.17), còn đưa 100 token cuối thì p(pos) = 0.407 (log-odds −0.38), tức dự đoán đổi về phía đúng, nên giả thuyết truncation được ủng hộ cho ca này (một ca, và p vẫn chưa chắc chắn). Số liệu tổng quát: accuracy 0.8791 với review ≤ 100 token so với 0.7973 với review dài hơn và 0.7385 với > 300 token; đây chỉ là tương quan vì độ dài lẫn với yếu tố khác.

### Ca 4 — `provided_csv:35024` (đúng: neg, p = 0.0085)
*"hobgoblins is a very cheap and badly done gremlins rip off … it was very painful to watch … avoid this one at all costs"* (51 token).

* **Token quan trọng (giả thuyết):** `cheap`, `badly`, `painful`, `avoid`.
* **Attention:** top-3 `cheap` (0.096), `painful` (0.047), `very` (0.042): trùng hai token ta cho là quan trọng.
* **Đối chiếu:** xóa `cheap` đưa log-odds −4.76 → −3.19; xóa `cheap, painful` → −2.20 (p 0.100, vẫn dự đoán âm). Xóa `avoid` gần như không đổi (−4.71). Đối chứng +0.002 ± 0.012. Tức hai token được attention chú ý quả có đóng góp, nhưng dự đoán còn lại nhờ các cue âm khác.
* **Nhận định:** Ca đúng, attention trùng với token mình cho là quan trọng: `cheap` và `painful` nằm trong top-3. Xóa `cheap, painful` đưa log-odds từ −4.76 lên −2.20 (p = 0.100) nhưng dự đoán vẫn âm, nên hai token này có đóng góp nhưng không phải bằng chứng duy nhất; `avoid` hầu như không ảnh hưởng (−4.71).

### Ca 5 — `provided_csv:6028` (đúng: pos, p = 0.9996)
*"for those that are great fans and collectors of dinosaurs … informative series … great models … fantastic photography … superb"* (42 token).

* **Token quan trọng (giả thuyết):** `great`, `informative`, `fantastic`, `superb`.
* **Attention:** top-3 `composition` (0.090), `our`, `great` (0.046); chỉ `great` là cue sentiment, mà đứng thứ 3. `composition` nhận nhiều attention nhất nhưng không phải từ cảm xúc.
* **Đối chiếu:** xóa `composition` chỉ giảm log-odds +7.93 → +7.29; xóa `great, fantastic, superb` còn +5.03 (p 0.994); đối chứng ≈ 0. Dự đoán đúng và rất bền, vì có nhiều cue dương dư thừa; top-1 attention không phải token quyết định.
* **Nhận định:** Ca đúng và rất chắc (p = 0.9996). Top-1 của attention là `composition`, không phải từ cảm xúc; chỉ `great` có mặt ở vị trí thứ 3. Xóa `composition` chỉ giảm log-odds +7.93 → +7.29, xóa `great, fantastic, superb` còn +5.03 (p = 0.994). Dự đoán bền vì review có nhiều cue dương dư thừa, và token nhận nhiều attention nhất không phải token quyết định.

### Ca 6 — `provided_csv:3286` (đúng: pos, p = 0.7203, kém chắc chắn)
*"i saw this movie when it aired on the wb and fell in love with riley smith … feel like being entertained and not much more …"* (51 token).

* **Token quan trọng (giả thuyết):** `fell in love`, `recommend`, `entertained`; `not much more` là sắc thái giảm nhẹ.
* **Attention:** top-3 `much`, `all`, `entertained`; chạm `entertained` và một phần của "not much more".
* **Đối chiếu:** xóa `entertained` hoặc `not much more` đều làm log-odds *tăng* (+0.95 → +1.59 / +1.44), ngược với kỳ vọng nếu `entertained` là cue dương. Đối chứng (xóa 3 token ngẫu nhiên) cũng *tăng* +0.104 ± 0.051; vì vậy các thay đổi ở ca này **nằm trong khoảng dao động của đối chứng** và không cho thấy bằng chứng rõ ràng nào về vai trò của các token đó.
* **Nhận định:** Ca đúng nhưng kém chắc chắn (p = 0.72) vì có "not much more". Xóa `entertained` hay `not much more` đều làm log-odds tăng (+0.95 → +1.59 / +1.44), ngược với kỳ vọng nếu `entertained` là cue dương; tuy nhiên xóa 3 token ngẫu nhiên cũng làm p tăng +0.104 ± 0.051, nên các thay đổi này nằm trong dao động của đối chứng và không cho bằng chứng rõ ràng về vai trò của các token đó.

## Giới hạn của kết luận

* Không kết luận "attention nhìn sai từ nên model sai". Ở các ca sai (1–3) chỉ có giả thuyết, được kiểm tra bằng xóa token; phép xóa không lật được dự đoán, ngoại lệ duy nhất là dùng 100 token cuối ở ca 3.
* Ở các ca đúng (4–6), token top-1 của attention không nhất quán với token đóng góp nhiều nhất (ca 5) hoặc thay đổi khi xóa không phân biệt được với nhiễu (ca 6): attention nhận được ≠ độ quan trọng đối với dự đoán.
* Sáu ca là mẫu nhỏ chọn tay; nhóm lỗi gán tay có thể sai và cần đọc lại toàn văn review để xác nhận.

## Kết luận

* **Nhóm lỗi cuối cùng:** ca 1 negation / ambiguous wording; ca 2 sarcasm (kèm noise); ca 3 truncation / long-range dependency.
* **Token quan trọng:** ca 1 `not worth` và `pleasantly surprised`; ca 2 `boring`, `pity`, `no fun` (chê) và `terrific` (gây nhầm); ca 3 phần đánh giá nằm ngoài 100 token đầu.
* **Giả thuyết được ủng hộ:** truncation (ca 3, đưa 100 token cuối thì p(pos) 0.991 → 0.407) và cue dương `terrific` kéo dự đoán (ca 2, log-odds +4.51 → +0.87). **Chưa được ủng hộ:** model xử lý "not worth" như phủ định (ca 1) và `boring`/`pity` là cue âm rõ ràng (ca 2).
