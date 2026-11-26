# LAB 10 — Calculations

## 1. Calculation Exercise 1 — Start / End (mục 6)

Context đã tokenize: `[The, capital, of, France, is, Paris, .]` (chỉ số từ 0), question `What is the capital of France?`, answer `Paris`.

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| token | The | capital | of | France | is | **Paris** | . |

- answer `Paris` ⇒ **start = 5, end = 5** (answer 1 token ⇒ start = end; điều kiện `i ≤ j` thoả).
- Đổi sang `The capital of France is Paris.` với answer `capital of France` ⇒ `capital` = 1, `of` = 2, `France` = 3 ⇒ **start = 1, end = 3**. `end` là chỉ số của token **cuối cùng bao gồm** (không phải exclusive): `answer = tokens[1:4]`.

**Chỉ số thấy bởi model.** Input thật là `[CLS] What is the capital of France ? [SEP] The capital of France is Paris . [SEP]` (17 token). Token context đầu tiên ở vị trí **9** (sau `[CLS]`, 7 token question, `[SEP]`) ⇒ trong `input_ids`, `Paris` ở **14** (= 9 + 5), `capital of France` ở **10–12** (= 9 + 1 … 9 + 3). Model dự đoán trên **toàn chuỗi**, nên nhãn `start_positions/end_positions` phải là chỉ số toàn chuỗi (kiểm chứng bằng `assert` trong `tokenizer_alignment.ipynb` mục 1).

## 2. Calculation Exercise 2 — Start loss (mục 7)

Gold start = 5, `P_start = [0.02, 0.05, 0.03, 0.10, 0.15, 0.60, 0.05]` (tổng = 1).

$$L_{start} = -\ln P_{start}(5) = -\ln 0.60 = \mathbf{0.5108}\ \text{nats}\ (=0.7370\ \text{bit}).$$

"Tương tự với end probability": answer `Paris` ⇒ end = 5. Đề không cho bảng `P_end`; **nếu dùng cùng bảng**: $L_{end} = 0.5108$ ⇒ $L = (0.5108 + 0.5108)/2 = \mathbf{0.5108}$. Với một bảng end khác tự đặt (`P_end(5) = 0.70`): $L_{end} = -\ln 0.70 = 0.3567$ ⇒ $L = (0.5108 + 0.3567)/2 = \mathbf{0.4338}$ (khớp `qm.qa_loss`, kiểm chứng trong notebook và `test_start_loss_exercise2`).

Mốc tham khảo: model đoán đều trên T vị trí cho loss $\ln T$ (T = 176 token trung bình ⇒ 5.17); xác suất đúng 0.20 ⇒ 1.609; 0.60 ⇒ 0.511; 0.99 ⇒ 0.010.

## 3. QA loss và vì sao cần **hai** prediction (mục 8)

$$L = \frac{L_{start} + L_{end}}{2},\quad L_{start} = -\log P_{start}(i),\ L_{end} = -\log P_{end}(j).$$

- **Một span cần hai bậc tự do** (vị trí và độ dài). Chỉ một prediction (start) không xác định span: cùng `start = 1` có thể là `capital`, `capital of`, `capital of France`. Cần thêm end (hoặc độ dài).
- **Hai softmax thay cho $T(T+1)/2$ lớp**: với T = 7 có 28 span hợp lệ nhưng chỉ cần 14 logit (`Linear(d, 2)` ⇒ `start_logits`, `end_logits` ∈ ℝ^T); phân tích $P(i,j) \approx P_{start}(i) P_{end}(j)$. Với T ≈ 384 là 74,000 span so với 768 logit.
- **Giải mã có ràng buộc**: vì hai phân phối độc lập nên có thể cho $j < i$ — với bảng ở notebook, **22.9%** khối xác suất tích rơi vào cặp không hợp lệ ($j<i$); giải mã chỉ xét $i \le j$ và $j - i + 1 \le 30$ (`qm.postprocess`).
- **Khác token classification**: NER gán nhãn cho *mọi* token (nhiều span), softmax trên **nhãn** tại mỗi vị trí; QA chỉ có **một** span, softmax trên **vị trí** (đúng một token là start, đúng một token là end).
- **Phân tích được lỗi theo biên**: hai head cho phép đếm riêng `#wrong start` và `#wrong end` (mục 25 của đề).

## 4. Calculation Exercise 3 — Character span (mục 12)

`The capital of France is Paris.` (độ dài 31): `The␣` = 0–3, `capital␣` = 4–11, `of␣` = 12–14, `France␣` = 15–21, `is␣` = 22–24, `Paris` = 25–29, `.` = 30.

1. index ký tự đầu của `Paris` = **25**;
2. index ký tự cuối = **29** (biên phải exclusive = **30**);
3. answer span theo character = **[25, 30)** (`context[25:30] == "Paris"`);
4. sau tokenization (`bert-base-cased`) token tương ứng là `Paris` với `offset_mapping = (25, 30)`;
5. **start token = 14**, 6. **end token = 14** (chỉ số trong chuỗi đầy đủ; **5** nếu chỉ đếm trong context).

Không nhầm ba loại chỉ số: **character** 25 ≠ **word** 5 (`Paris.` dính dấu chấm khi `split()`) ≠ **token** 5 (trong context) / 14 (toàn chuỗi). Căn chỉnh tổng quát: start_token = token có `start ≤ answer_start < end`, end_token = token có `start < answer_end ≤ end` (answer_end exclusive), **chỉ xét token context**; nếu answer không nằm trọn trong context của window ⇒ nhãn `[CLS] = (0, 0)`.

## 5. Sliding window (mục 13)

Gọi $q$ = số token question (không tính special), $N$ = số token context, $L$ = `max_length`, $S$ = `stride` (số token chồng nhau). Dung lượng context mỗi window $c = L - q - 3$ (trừ `[CLS]`, 2 `[SEP]`); bước trượt $c - S$:

$$n_{windows} = 1 \ (N \le c),\qquad n_{windows} = 1 + \left\lceil \frac{N - c}{c - S} \right\rceil \ (N > c).$$

Ví dụ đề: window dài 384, stride 128 ⇒ bước 256: `[0:384]`, `[256:640]`, `[512:896]`; hai window kề nhau chồng 128 token. Kiểm chứng trên một câu của train (q = 16, N = 591): `L = 128, S = 42` ⇒ c = 109, bước 67 ⇒ $1 + \lceil 482/67 \rceil$ = **9** window (tokenizer trả đúng 9); `L = 256, S = 85` ⇒ c = 237 ⇒ **4**; `L = 384, S = 128` ⇒ c = 365 ⇒ **2**. Công thức khớp tokenizer trên 500 câu × 2 cấu hình (0 sai lệch).

## 6. EM và token-F1 tính tay (mục 18–19)

Normalize: lowercase → bỏ dấu câu → bỏ `a/an/the` → gom khoảng trắng. F1 dùng **overlap multiset** (`Counter`).

| Gold | Prediction | sau normalize | #overlap | P | R | F1 | EM |
|---|---|---|---|---|---|---|---|
| `Paris France` | `France` | `paris france` / `france` | 1 | 1/1 = 1 | 1/2 = 0.5 | 2·1·0.5/1.5 = **2/3 = 0.667** | 0 |
| `Paris` | `the city of Paris` | `paris` / `city of paris` | 1 | 1/3 | 1/1 = 1 | 2·(1/3)·1/(4/3) = **0.5** | 0 |
| `New York City` | `New York` | 3 / 2 token | 2 | 2/2 = 1 | 2/3 | 2·1·(2/3)/(5/3) = **0.8** | 0 |
| `Paris` | `paris.` | `paris` / `paris` | 1 | 1 | 1 | **1** | 1 |
| `x b b` (pred) | `b b b` (gold) | — | 2 (multiset) | 2/3 | 2/3 | **2/3** | 0 |

Vì vậy "dự đoán có F1 > 0 nhưng EM = 0" xảy ra khi prediction và gold chia sẻ ít nhất một token sau normalize nhưng không trùng hoàn toàn (thừa/thiếu từ). Khi SQuAD có nhiều gold, EM/F1 lấy **max** trên các gold. Tất cả giá trị trên được `assert` trong `test_qa_model.py` (`test_f1_paris_france_example`, `test_f1_multiset_overlap`, `test_exact_match_strict`).

## 7. Shape (mục 15)

$B = 8,\ T = 384,\ d = 768$: `input_ids/attention_mask/token_type_ids` $(8, 384)$ → $H = $ encoder$(x)$ $(8, 384, 768)$ → `Linear(768, 2)` ⇒ $Z\ (8, 384, 2)$ → `start_logits`, `end_logits` $(8, 384)$ → softmax theo $T$ (mỗi hàng tổng = 1) → loss `CrossEntropy(logits[B,T], positions[B])`. Head chỉ có $768 \times 2 + 2 = 1{,}538$ tham số trên 107,721,218 của toàn model. Padding được mask (`−1e4` trước softmax) nên không nhận xác suất (`test_qamodel_shapes_and_padding_invariance`).
