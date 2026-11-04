# LAB 07 — Transformer Architecture (IMDb)

Transformer encoder block tự ghép (không dùng `nn.TransformerEncoderLayer` / `nn.MultiheadAttention` trong core), PE sin/cos tự viết, 6 experiment + application + error analysis. Dữ liệu, split, vocabulary **tái dùng đúng của LAB 05**.

## Cách chạy

```powershell
cd Lab07
python positional_encoding.py          # unit test PE d=4, pos 0..2
python transformer_block.py            # shape B=8,T=100,d=128,h=4 + padding mask
# chạy notebook (kernel python3, cwd = Lab07; ~31 phút trên RTX 3050 6GB)
python -m nbconvert --to notebook --execute --inplace transformer_classification.ipynb
```

- Cần: PyTorch (CUDA), pandas, numpy, scikit-learn, matplotlib, jupyter. Python 3.14, torch 2.12.0+cu126.
- Notebook đọc `../Lab05/data/{train,test}.csv` và `../Lab05/sequence_models.py` (split 31,879 / 7,904 / 10,000, vocab 30,000, seed 42 — có `assert` khớp LAB 05). LSTM baseline cũng là `SequenceClassifier` của LAB 05.
- **Windows:** phải `import pandas` *trước* `torch` (nếu không process crash `0xC0000374`) — đã xử lý trong cell setup.
- `LAB07_FAST=1` là cờ debug (1 epoch, 1 seed); mặc định tắt, kết quả nộp chạy với cờ tắt.

## Cấu hình chung

d_model 128, d_ff 256, 2 layer, dropout 0.1, batch 128, Adam lr 1e-3, 5 epoch, max_length 128 (trừ Exp 5), keep-first truncation, chọn checkpoint theo val loss, đánh giá trên test. LSTM: embedding 128, hidden 128 (cùng split/vocab/batch/optimizer/lr/epoch). Seed 42; các cấu hình chính chạy thêm seed 43, 44 (mean ± std mẫu trong `extra` của `results.csv`). Thời gian train gồm cả validation mỗi epoch.

## Kết quả chính (test, seed 42 | mean ± std 3 seeds)

| Thí nghiệm | Kết quả |
|---|---|
| Exp 1 permutation | Không PE: S2 là hoán vị hàng của S1, pooled khác ≤ 1.8e-7. Có PE: 0.157–0.250. IMDb: bỏ PE không giảm (acc 0.8391 ± 0.0054 vs 0.8353 ± 0.0050) |
| Exp 2 heads 1/2/4/8 (d_k 128/64/32/16) | params đều 4,105,089; acc 3 seeds 0.8356 / 0.8321 / 0.8353 / 0.8331 (std 0.003–0.007); thời gian 42.8 / 52.7 / 63.7 / 85.7 s |
| Exp 3 head diversity | cos giữa head 0.203–0.253 (layer 1), 0.061–0.175 (layer 2); zero 1 head layer 2: −0.09…0.28 điểm acc val |
| Exp 4 LSTM vs Transformer | LSTM acc 0.8228 / F1 0.8198 (3 seeds F1 0.8216 ± 0.0026) vs Transformer 0.8382 / 0.8349 (0.8339 ± 0.0089); params 3,972,225 vs 4,105,089; 60.9 vs 63.7 s |
| Exp 5 T = 32/64/128/256 | tokens giữ lại 13.65/27.02/51.07/76.79%; F1 Tf 0.7435/0.7992/0.8349/0.8668; thời gian 37.2/40.6/63.7/161.5 s; peak CUDA 353/474/518/1,438 MB |
| Exp 6 pooling | Mean 0.8353 ± 0.0050 vs CLS 0.8331 ± 0.0023 (acc, 3 seeds) — không khác biệt đo được |

Đầy đủ trong `results.csv` (71 dòng; cột `extra` là JSON).

## Files

`positional_encoding.py`, `transformer_block.py` (MultiHeadSelfAttention, TransformerBlock, TransformerClassifier + self-test), `transformer_classification.ipynb` (đã chạy, có output), `results.csv`, `positional_encoding.png`, `attention_heads.png`, `learning_curves.png`, `confusion_matrix.png`, `prediction.md`, `error_analysis.md`, `reflection.md`.

## Giả định, hạn chế, việc còn thiếu

- `prediction.md` do sinh viên cung cấp, nội dung giữ nguyên; file được đưa vào repo **sau khi** các experiment đã chạy xong (không phải trước). Cột Đúng/Sai trong bảng prediction (cuối notebook) là đối chiếu với số liệu, chưa được sinh viên xác nhận.
- Đề không cho số cho X, W_Q/K/V (mục 6) và W₂, b₂ (mục 14) → dùng số tự chọn (ghi trong notebook).
- Đúng 3 seed, 1 split; chênh lệch ≤ 1–2 điểm giữa heads/pooling/PE nằm trong nhiễu. So sánh LSTM vs Transformer chưa tách từng thành phần kiến trúc (khác cả pooling).
- Exp 1 dùng trọng số ngẫu nhiên (chỉ chứng minh khả năng phân biệt thứ tự, không phải học được).
- Peak CUDA đo trong lúc train (gồm tham số, optimizer, validation); "attention memory lý thuyết" = B·h·T²·4 byte cho một ma trận scores của một layer.
- Model checkpoint không lưu ra đĩa (train lại trong notebook). Không có kiểm tra đồng thời nhiều head bị ablate; Exp 3 đo 500 review test.
- Máy: RTX 3050 Laptop 6GB; số liệu thời gian phụ thuộc máy.

## AI assistance statement

- **Tool:** Claude Code (model Claude Sonnet 5.5).
- **Purpose:** viết code, chạy experiment, soạn toàn bộ tài liệu theo yêu cầu của sinh viên.
- **Generated:** toàn bộ code (`positional_encoding.py`, `transformer_block.py`, notebook) và toàn bộ văn bản: `calculations.md` (tính toán, trả lời các câu hỏi của đề, bảng so sánh LAB 01→07 và kiểm chứng prediction; file này đã gỡ khỏi repo, các ô trả lời tương ứng nằm trong notebook), `error_analysis.md`, `reflection.md`, README. Nhiều phần trong số này (calculation, shape/PE calculation, giải thích multi-head, diễn giải attention heads, error analysis, reflection) thuộc nhóm mục 30 của đề *không cho phép* AI thay thế. `prediction.md` do sinh viên cung cấp, AI không viết.
- **Modified:** chưa ghi nhận chỉnh sửa nội dung nào của sinh viên đối với các file trên tại thời điểm commit này. *(Nếu sinh viên sửa sau, ghi cụ thể file và phần đã sửa tại đây.)*
- **Verified:** unit test PE d=4 pos 0–2 (`atol 5e-5`); `check_shapes` B=8,T=100,d=128,h=4 (Q,K,V=(8,4,100,32), scores=(8,4,100,100), output=(8,100,128)) với `assert`; test padding mask (nội dung PAD không ảnh hưởng vị trí hợp lệ; attention tới PAD = 0); so sánh số học với `nn.TransformerEncoderLayer` cùng trọng số (sai khác tối đa 4.77e-07 không mask, 7.15e-07 có padding mask); split/vocab khớp LAB 05 bằng `assert`; notebook chạy từ đầu đến cuối không lỗi (1,890 s). Các số trong `calculations.md` (trước khi gỡ khỏi repo), `error_analysis.md`, `reflection.md` được đối chiếu với `results.csv` (104 giá trị khớp); riêng các số của 6 mẫu lỗi lấy từ output notebook, chưa đối chiếu tự động. **Chưa được xác minh / sinh viên tự xác nhận:** (1) cột Đúng/Sai của bảng prediction; (2) các nhận định nguyên nhân trong `error_analysis.md` (chỉ là giả thuyết, chưa can thiệp xoá/thay token); (3) phần so sánh với BERT trong `reflection.md` (kiến thức chung, không đo trong lab này); (4) số liệu LAB 05 được trích lại (TF-IDF 0.8958, GRU F1 0.7659 → 0.8433) lấy từ README LAB 05.
