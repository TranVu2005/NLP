# LAB 07 — Tính toán & trả lời câu hỏi của đề

> Mọi số liệu lấy từ output của `transformer_classification.ipynb` / `results.csv` (seed 42, trừ khi ghi "3 seeds" = seed 42/43/44, mean ± std mẫu).
> Đề **không cho số cụ thể** cho `X, W_Q, W_K, W_V` (mục 6) và `W_2, b_2` (mục 14) → các giá trị dưới đây là **tự chọn**, ghi rõ để chấm lại bằng tay.

## A. Tính toán

### Mục 6 — một attention head (d_k = 2)

```
X  = [[1,0,1,0],[0,2,0,1],[1,1,1,1]]      (3×4)
W_Q = [[1,0],[0,1],[1,0],[0,1]]   W_K = [[0,1],[1,0],[0,1],[1,0]]   W_V = [[1,0],[0,1],[0,1],[1,0]]   (4×2)
```

| | giá trị |
|---|---|
| Q = XW_Q | [[2,0],[0,3],[2,2]] |
| K = XW_K | [[0,2],[3,0],[2,2]] |
| V = XW_V | [[1,1],[1,2],[2,2]] |
| S = QKᵀ/√2 | [[0, 4.2426, 2.8284],[4.2426, 0, 4.2426],[2.8284, 4.2426, 5.6569]] |
| A = softmax(S) (theo hàng) | [[0.0114, 0.7952, 0.1933],[0.4964, 0.0071, 0.4964],[0.0454, 0.1867, 0.7679]] |
| Z = AV | [[1.1933, 1.9886],[1.4964, 1.5036],[1.7679, 1.9546]] |

Ví dụ tay: hàng 1 của QKᵀ = [2,0]·[0,2], [2,0]·[3,0], [2,0]·[2,2] = 0, 6, 4 → chia √2 = 0, 4.2426, 2.8284 → softmax = [0.0114, 0.7952, 0.1933] (tổng = 1; code kiểm tra tổng mỗi hàng = 1). Đây đúng là attention của LAB 06, chỉ khác là Q, K, V đến từ ba ma trận chiếu.

### Mục 7 — vì sao concat trước W^O

Với 2 head, mỗi head ∈ ℝ^{T×2}, concat ∈ ℝ^{T×4}. Ví dụ số (head 2 dùng W đã đảo cột của head 1 — chỉ để minh hoạ đại số; head 2 này có attention giống hệt head 1 nên **không** minh hoạ đa dạng pattern):

concat = [[1.1933,1.9886,1.9886,1.1933],[1.4964,1.5036,1.5036,1.4964],[1.7679,1.9546,1.9546,1.7679]]; với W^O (4×4) tự chọn, `concat·W^O = head₁·W^O[:2] + head₂·W^O[2:]` (code: `True`).

Trả lời:
1. Concat đưa các head về **một tensor T×(h·d_k) = T×d_model**; W^O ∈ ℝ^{d_model×d_model} chiếu về đúng d_model để cộng residual `x + MHSA(x)` và đưa vào block kế tiếp.
2. Đẳng thức trên cho thấy W^O là h khối hàng, mỗi khối chiếu đầu ra của một head: **W^O là nơi thông tin của các head được trộn và gán trọng số học được**. Trước W^O, mỗi head chỉ nhìn subspace d_k riêng của nó; không concat thì không có chỗ nào để kết hợp chúng.

### Mục 10 — Positional encoding, d = 4, pos = 0,1,2

PE(pos,2i) = sin(pos/10000^{2i/d}), PE(pos,2i+1) = cos(pos/10000^{2i/d}). Với d=4: i=0 → tần số 1; i=1 → 10000^{2/4} = 100 → tần số 0.01.

| pos | [sin(pos), cos(pos), sin(pos/100), cos(pos/100)] | tính tay | output `positional_encoding.py` |
|---|---|---|---|
| 0 | [sin0, cos0, sin0, cos0] | [0, 1, 0, 1] | [0, 1, 0, 1] |
| 1 | [sin1, cos1, sin0.01, cos0.01] | [0.8415, 0.5403, 0.0100, 0.99995] | [0.84147, 0.5403, 0.01, 0.99995] |
| 2 | [sin2, cos2, sin0.02, cos0.02] | [0.9093, −0.4161, 0.0200, 0.9998] | [0.9093, −0.41615, 0.02, 0.9998] |

Khớp (unit test `atol=5e-5` pass). Mỗi vị trí có một vector khác nhau; các chiều đầu dao động nhanh, chiều sau chậm (heatmap `positional_encoding.png`, panel a).

### Mục 14 — FFN

x = [1,2]ᵀ, W₁ = I, b₁ = [0,−1]ᵀ → W₁x + b₁ = [1,1] → h = ReLU = [1,1]. Đề không cho W₂, b₂; chọn W₂ = [[1,−1],[2,0]], b₂ = [0,1] → y = W₂h + b₂ = [0, 3]. FFN chỉ là biến đổi phi tuyến **độc lập trên từng vị trí** (không trộn token với nhau); trộn token là việc của attention.

### Mục 15 — Residual: F(x) so với x + F(x)

- Chỉ dùng F(x): đầu ra thay thế hoàn toàn đầu vào; Jacobian của chồng L lớp là tích L ma trận → co nhỏ/biến mất.
- Dùng x + F(x): Jacobian mỗi lớp là I + ∂F/∂x → luôn có "đường thẳng" cho thông tin và gradient đi qua.

Thí nghiệm đồ chơi (notebook mục 3, F(x)=tanh(Wx), W~N(0, 0.5²/64), d=64, seed 0; **không** phải mô hình đã huấn luyện): chuẩn Frobenius của ∂out/∂x:

| số lớp | chỉ F | x + F |
|---|---|---|
| 1 | 3.304 | 8.703 |
| 5 | 2.037e-1 | 11.58 |
| 10 | 6.278e-3 | 16.11 |
| 20 | 6.527e-6 | 27.06 |
| 40 | 5.734e-12 | 58.69 |

Chỉ-F: gradient suy giảm theo cấp số nhân; x+F: không biến mất (nhưng độ lớn tăng dần → đó là lý do cần LayerNorm sau mỗi cộng). **LayerNorm** chuẩn hoá từng vector token theo chiều feature (mean 0, var 1, rồi γ, β học được) → giữ scale ổn định; block của tôi là post-norm: `LN(x + Attn(x))` rồi `LN(x + FFN(x))`.

### Mục 18 — bảng shape (B=8, T=100, d_model=128, h=4, d_k=32)

Từ `check_shapes()` (có `assert` từng bước; FFN hidden = 512 vì test dùng d_ff = 4·d_model; trong các experiment d_ff = 256):

| bước | shape |
|---|---|
| X | (8, 100, 128) |
| Q, K, V (sau split heads) | (8, 4, 100, 32) |
| scores = QKᵀ/√d_k | (8, 4, 100, 100) |
| attention weights (softmax) | (8, 4, 100, 100) |
| head outputs = A·V | (8, 4, 100, 32) |
| concat heads | (8, 100, 128) |
| sau W^O | (8, 100, 128) |
| Add&Norm 1 | (8, 100, 128) |
| FFN hidden / FFN out | (8, 100, 512) / (8, 100, 128) |
| output | (8, 100, 128) |

Ma trận attention là T×T vì mỗi trong T query chấm với mỗi trong T key. Block (d=128, h=4, d_ff=256) có 132,480 tham số. So với `nn.TransformerEncoderLayer` (post-norm, cùng trọng số, dropout 0): sai khác tối đa 4.77e-07 (không mask) và 7.15e-07 (padding mask, vị trí hợp lệ).

---

## B. Trả lời câu hỏi của đề

### Mục 11–12 — vì sao thứ tự là vấn đề khi không có PE

Self-attention không có khái niệm vị trí: hoán vị hàng của X chỉ hoán vị hàng của đầu ra (permutation-equivariant). Thực nghiệm (Exp 1, d=8, 2 head, trọng số ngẫu nhiên seed 42; `positional_encoding.png`):

| cặp | module | PE | S2 là hoán vị hàng của S1? | max lệch khỏi hoán vị | ‖mean(S1) − mean(S2)‖ |
|---|---|---|---|---|---|
| I love NLP / NLP love I | attention | không | **True** | 1.2e-07 | 1.1e-07 |
| | attention | có | False | 0.389 | 0.157 |
| | full block | không | **True** | 2.4e-07 | 1.8e-07 |
| | full block | có | False | 1.300 | 0.250 |
| dog bites man / man bites dog | attention | không | **True** | 3.0e-08 | 7.3e-08 |
| | attention | có | False | 0.146 | 0.223 |
| | full block | không | **True** | 1.8e-07 | 6.1e-08 |
| | full block | có | False | 0.845 | 0.194 |

Không PE: representation của từng token giống hệt ở cả hai câu (cosine = 1.0), sau mean-pooling hai câu khác nhau ~1e-7 (sai số float) → mô hình **không thể** phân biệt "dog bites man" với "man bites dog". Có PE: cùng token ở vị trí khác có representation khác (cosine 0.85–0.9998) và pooled khác nhau 0.19–0.25.
Hạn chế: trọng số ngẫu nhiên, chỉ chứng minh *khả năng* phân biệt, không chứng minh mô hình học được ý nghĩa của thứ tự.
**Lưu ý quan trọng:** trên IMDb, bỏ PE không làm giảm kết quả (mean-pool, 4 head, 3 seeds): acc 0.8391 ± 0.0054 (không PE) so với 0.8353 ± 0.0050 (có PE); F1 0.8413 ± 0.0007 so với 0.8339 ± 0.0089. Sự cần thiết của PE được chứng minh ở bài permutation, **không** ở task cảm xúc này (cảm xúc chủ yếu do từ vựng quyết định; BoW-LR đạt 0.885 trên test).

### Mục 19 — số heads (d_model = 128)

| heads | d_k | params | thời gian train (s, seed 42) | acc | F1 | acc 3 seeds | F1 3 seeds |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 4,105,089 | 42.8 | 0.8337 | 0.8378 | 0.8356 ± 0.0045 | 0.8366 ± 0.0059 |
| 2 | 64 | 4,105,089 | 52.7 | 0.8349 | 0.8292 | 0.8321 ± 0.0071 | 0.8242 ± 0.0085 |
| 4 | 32 | 4,105,089 | 63.7 | 0.8382 | 0.8349 | 0.8353 ± 0.0050 | 0.8339 ± 0.0089 |
| 8 | 16 | 4,105,089 | 85.7 | 0.8353 | 0.8364 | 0.8331 ± 0.0026 | 0.8335 ± 0.0036 |

- *Tăng số heads có tăng representation dimension không?* **Không.** Q/K/V/W^O đều là d_model×d_model nên #params giữ nguyên (4,105,089 ở mọi cấu hình) và concat luôn là h·d_k = 128. Heads chỉ **chia** không gian 128 chiều thành h subspace.
- *d_k khi d_model cố định:* d_k = d_model/h: 128 → 64 → 32 → 16.
- Accuracy/F1 không tăng đơn điệu và chênh lệch nằm trong độ lệch chuẩn giữa các seed → tăng heads **không** chắc chắn tăng accuracy. Thời gian tăng (mean 3 seeds: 45.4 → 53.3 → 64.5 → 84.6 s): nhiều head nhỏ hơn, tensor 4 chiều lớn hơn.

### Mục 20 — head có tạo pattern khác nhau không?

Mô hình 4 head, 2 layer, 500 review test (`attention_heads.png`; cos = cosine giữa attention map của hai head, trung bình trên doc; H̃ = entropy chuẩn hoá, 1 = đều):

| layer | H̃ của 4 head | cos giữa các head (min–max) | zero 1 head: giảm acc val (điểm %) |
|---|---|---|---|
| 1 | 0.84 / 0.83 / 0.83 / 0.77 | 0.203–0.253 | 0.76 / 0.76 / 0.71 / 0.40 |
| 2 | 0.68 / 0.57 / 0.54 / 0.39 | 0.061–0.175 | −0.09 / 0.11 / 0.28 / 0.10 |

- Các head **không** học cùng một pattern (cos ≤ 0.25 ở layer 1, ≤ 0.175 ở layer 2). Layer 1 phân tán hơn (H̃ ≈ 0.8); head cuối layer 2 tập trung nhất (H̃ = 0.39). Trên hình, nhiều head layer 2 dồn gần như toàn bộ query vào **một cột key** (vd. S2 layer 2 head 3: hầu hết query → "enjoyed", H̃ = 0.05).
- Bỏ riêng một head layer 2 gần như không đổi val acc (−0.09 đến 0.28 điểm; val có 7,904 mẫu) → từng head layer 2 **gần như thừa riêng lẻ** về mặt acc; không phân biệt được "thừa thật" với "residual path bù lại" (chưa ablate đồng thời nhiều head). Layer 1 nhạy hơn (0.4–0.76 điểm).
- Không gán nhãn syntax/semantics: không có bằng chứng ngoài các attention map này.

### Mục 21 — LSTM vs Transformer (cùng split, vocab, embedding 128, batch 128, Adam 1e-3, 5 epochs, T=128)

| model | params (không-embedding) | train time (s) | acc | precision | recall | F1 | acc 3 seeds | F1 3 seeds |
|---|---|---|---|---|---|---|---|---|
| LSTM (hidden 128) | 3,972,225 (132,225) | 60.9 | 0.8228 | 0.8339 | 0.8062 | 0.8198 | 0.8188 ± 0.0054 | 0.8216 ± 0.0026 |
| Transformer (2 layer, 4 head, d_ff 256, mean) | 4,105,089 (265,089) | 63.7 | 0.8382 | 0.8523 | 0.8182 | 0.8349 | 0.8353 ± 0.0050 | 0.8339 ± 0.0089 |

Transformer hơn ~1.2–1.7 điểm (mean 3 seeds) — khoảng 2–3 độ lệch chuẩn, chỉ 3 seed nên là xu hướng, không phải kết luận chắc. Cả hai thấp hơn TF-IDF+LR của LAB 05 (acc 0.8958, cùng split) vì chỉ giữ 51% token (T=128) và huấn luyện từ đầu.
Các thành phần *có thể* gây khác biệt (giả thuyết, **chưa tách bằng ablation**): (i) tương tác token–token trực tiếp (đường đi 1 bước, LSTM phải truyền qua tối đa 128 bước); (ii) pooling: mean trên mọi vị trí (gradient chạm mọi token) so với trạng thái cuối của LSTM — hai mô hình khác pooling nên không quy được riêng cho attention; (iii) residual + LayerNorm giúp tối ưu: val loss LSTM còn giảm ở epoch 5 (best epoch = 5 cả 3 seed) còn Transformer đạt tốt nhất ở epoch 3–5 rồi overfit; (iv) số tham số không-embedding của Transformer gấp đôi.

### Mục 22 — max_length

| T | tokens giữ lại | docs bị cắt | T² | attention matrix/layer, lý thuyết (B=128,h=4,fp32) | peak CUDA khi train (đo) | thời gian Tf (s) | F1 Tf | F1 LSTM |
|---|---|---|---|---|---|---|---|---|
| 32 | 13.65% | 99.72% | 1,024 | 2 MB | 353 MB | 37.2 | 0.7435 | 0.7470 |
| 64 | 27.02% | 94.92% | 4,096 | 8 MB | 474 MB | 40.6 | 0.7992 | 0.7832 |
| 128 | 51.07% | 74.96% | 16,384 | 32 MB | 518 MB | 63.7 | 0.8349 | 0.8198 |
| 256 | 76.79% | 29.27% | 65,536 | 128 MB | 1,438 MB | 161.5 | 0.8668 | 0.8587 |

(Lý thuyết = B·h·T²·4 byte cho **một** ma trận scores của **một** layer; peak đo gồm tham số, optimizer, activations, softmax, dropout… nên không bằng.) Một seed (42) mỗi T.
128 → 256: phần tử attention 16,384 → 65,536 (×4); thời gian train 63.7 → 161.5 s (×2.5), peak 518 → 1,438 MB (×2.8), trong khi F1 chỉ +0.032. Micro-benchmark fwd+bwd (B=16): T=512 → 1024: 46.3 → 152.9 ms (×3.3), peak 910 → 2,184 MB; LSTM 78.8 → 135.1 ms, 488 → 532 MB — ở T=1024 Transformer chậm hơn LSTM (152.9 vs 135.1 ms) và dùng nhiều bộ nhớ gấp ~4. Ngoại suy: T=1024 cần 128·4·1024²·4 B = 2,048 MB chỉ cho **một** ma trận scores của **một** layer.
*Vì sao không giải quyết mọi thứ bằng tăng length:* chi phí O(T²) về thời gian và bộ nhớ (số liệu trên); lợi ích F1 giảm dần theo chi phí; ngay ở T=256 vẫn có 29% review bị cắt; PE/độ dài tối đa cố định lúc huấn luyện.

### Mục 25 — pooling

| pooling | acc (seed 42) | F1 (seed 42) | thời gian (s) | acc 3 seeds | F1 3 seeds |
|---|---|---|---|---|---|
| Mean (mask padding) | 0.8382 | 0.8349 | 63.7 | 0.8353 ± 0.0050 | 0.8339 ± 0.0089 |
| CLS-style (token [CLS] học được ở đầu) | 0.8307 | 0.8365 | 72.5 | 0.8331 ± 0.0023 | 0.8347 ± 0.0037 |

Chênh lệch ≤ 0.25 điểm giữa hai cách, nhỏ hơn độ lệch chuẩn → **không thấy ảnh hưởng đo được** trong thí nghiệm này (3 seeds, 5 epochs, huấn luyện từ đầu). CLS chậm hơn (T+1 token) và chỉ thêm 128 tham số.

### Mục 27 — từ LAB 01 đến LAB 07

```
TF-IDF
  ↓ Limitation: vector thưa, chiều cao; túi từ — mất thứ tự từ; "good" và "great" là hai chiều không liên quan
Word2Vec (trung bình)
  ↓ Limitation: mỗi từ một vector tĩnh (không theo ngữ cảnh, đa nghĩa); trung bình vẫn là túi từ, mất thứ tự
LSTM
  ↓ Limitation: tính tuần tự theo timestep (không song song theo T); thông tin phải qua hidden state kích thước cố định;
    phụ thuộc xa khó (LAB 05, GRU: F1 0.7659 → 0.8433 khi tăng length 50 → 200)
Attention
  ↓ Limitation: nếu chỉ thêm lên RNN thì backbone vẫn tuần tự; bản thân attention không biết thứ tự (cần PE); chi phí O(T²)
Transformer
  ↓ Remaining limitation: O(T²) thời gian/bộ nhớ (đo ở Exp 5: T=1024 tốn 2,184 MB so với LSTM 532 MB), độ dài tối đa cố định;
    huấn luyện từ đầu cần nhiều dữ liệu (acc 0.835 < TF-IDF 0.8958 của LAB 05); attention ≠ giải thích → LAB 08: pretraining/BERT
```

### Mục 28 — kiểm chứng prediction

Cột "Dự đoán" copy nguyên từ `prediction.md` (không sửa).

| # | Dự đoán (từ `prediction.md`) | Kết quả (experiment) | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | **Có.** Self-attention tính mọi cặp token bằng phép nhân ma trận cùng lúc; RNN phải đợi h_{t-1}. (90%) | Benchmark fwd+bwd B=16: T=128: Tf 16.7 vs LSTM 25.0 ms; T=256: 17.1 vs 39.4 ms (LSTM tăng ×1.6, Tf gần như phẳng). Nhưng T=1024: 152.9 vs 135.1 ms; train thực (gồm validation) Tf chậm hơn LSTM ở cả T=32/64/128/256 (37.2 vs 31.5; 40.6 vs 37.7; 63.7 vs 60.9; 161.5 vs 93.5 s) | **Đúng** (về tính song song) | Bằng chứng ủng hộ: step time Tf gần như không đổi khi T gấp đôi trong khi LSTM tăng. Lưu ý: "song song hơn" không đồng nghĩa "thời gian train ngắn hơn" — LSTM dùng cuDNN tối ưu và Tf bị O(T²) ở T lớn. |
| 2 | **Không.** Self-attention (không mask) là permutation-equivariant; embedding không chứa vị trí. (90%) | Exp 1: không PE, S2 là hoán vị hàng của S1 (lệch tối đa 2.4e-7); cosine cùng token ở hai câu = 1.0 | **Đúng** | Khớp đúng lý do đã nêu. |
| 3 | **Không.** Output từng token chỉ bị hoán vị; sau mean pooling hai câu cho cùng vector. (90%) | Không PE: ‖mean(S1) − mean(S2)‖ = 1.1e-7 / 1.8e-7 (I love NLP) và 7.3e-8 / 6.1e-8 (dog bites man), tức bằng 0 trong sai số float; có PE: 0.157–0.250 | **Đúng** | Chính xác như dự đoán, kể cả chi tiết "output chỉ bị hoán vị" (`S2_is_row_permutation_of_S1 = True`). |
| 4 | **Không bắt buộc**; thường khác nhưng có thể có head gần như thừa. (80%) | cos giữa attention map các head: 0.203–0.253 (layer 1), 0.061–0.175 (layer 2). Zero một head layer 2 chỉ giảm acc val −0.09…0.28 điểm; layer 1: 0.40–0.76 điểm | **Đúng** (phần "head thừa" được ủng hộ một phần) | Các head khác pattern. Head layer 2 gần như dispensable riêng lẻ, nhưng chưa phân biệt được "thừa thật" với "residual bù" (chưa ablate đồng thời nhiều head). |
| 5 | **Không.** d_model cố định nên d_k giảm, số tham số gần như không đổi; có thể bão hoà/dư thừa. (85%) | acc 3 seeds heads 1/2/4/8: 0.8356 / 0.8321 / 0.8353 / 0.8331 (std 0.003–0.007); params đều 4,105,089 | **Đúng** | Số tham số không chỉ "gần như" mà **chính xác** không đổi (Q/K/V/W^O đều d_model×d_model). Accuracy không tăng đơn điệu, chênh lệch trong nhiễu. |
| 6 | **Không.** Chi phí O(T²); T ngắn/model nhỏ thì overhead nhỏ, T rất dài thì bộ nhớ attention tăng nhanh. (80%) | Train thực: Tf không nhanh hơn ở T nào. Benchmark: Tf nhanh hơn ở T=64–512, chậm hơn ở T=1024 (152.9 vs 135.1 ms); peak 2,184 MB vs 532 MB. T=128→256: attention ×4, peak train 518→1,438 MB | **Đúng** | Kết luận "không luôn nhanh hơn" và phần "T dài thì bộ nhớ tăng nhanh" đều được xác nhận. Phần lý do về T ngắn không khớp hẳn: ở T=32 Tf (huấn luyện thực) vẫn chậm hơn LSTM chứ không "overhead nhỏ" mang lại lợi thế. |
