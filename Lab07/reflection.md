# Reflection — từ Transformer (LAB 07) đến BERT (LAB 08)
## 1. Lab này cho thấy gì

- Ghép được block: MHSA → Add&LayerNorm → FFN → Add&LayerNorm; khớp `nn.TransformerEncoderLayer` tới 7e-07 khi cùng trọng số.
- PE cần thiết để phân biệt thứ tự (Exp 1: không PE thì "dog bites man" và "man bites dog" cho pooled representation khác nhau ~1e-7; có PE thì 0.19–0.25) — nhưng trên IMDb bỏ PE không làm kết quả tệ hơn (acc 0.8391 vs 0.8353, 3 seeds). Bài học: một thành phần cần về mặt kiến trúc chưa chắc đo được tác dụng trên mọi task.
- Transformer từ đầu đạt acc 0.8353 ± 0.0050 (3 seeds), hơn LSTM (0.8188 ± 0.0054) ~1.6 điểm nhưng **thấp hơn TF-IDF + LR của LAB 05 (0.8958)**; tăng số head không giúp (params không đổi, kết quả trong nhiễu); chi phí tăng theo T² (T=256: 161.5 s, 1,438 MB so với T=128: 63.7 s, 518 MB).

## 2. Vì sao cần pretraining

Mô hình ở đây học từ đầu trên ~32k review: riêng bảng embedding đã có 30,000×128 = 3.84M tham số trong tổng 4.1M, được học chỉ từ tín hiệu nhãn cảm xúc; train loss giảm xuống 0.25 trong khi val loss tăng lại từ epoch 3–4 (overfit). Thiếu dữ liệu nhãn nên không học được cú pháp, đồng tham chiếu, kiến thức thế giới — các lỗi ở `error_analysis.md` (từ vựng miền chuyên biệt, giọng mỉa/đối lập, "okay, I guess" → P(pos)=0.0038) là những chỗ pretraining trên văn bản không nhãn khổng lồ (masked language modeling) có thể giúp: mô hình mang vào fine-tuning biểu diễn ngữ cảnh đã học, chỉ cần ít nhãn cho task.

## 3. Các thành phần của lab này nằm ở đâu trong BERT

| thành phần | LAB 07 (của tôi) | BERT (dự kiến, kiểm tra ở LAB 08) |
|---|---|---|
| Input | embedding từ + PE sin/cos cố định | tổng token + position + segment embedding; position embedding **học được** |
| [CLS] | thử "CLS-style": vector học được ở đầu chuỗi; không hơn mean-pool (0.8331 ± 0.0023 vs 0.8353 ± 0.0050) vì không được pretrain | token đặc biệt đầu chuỗi; trạng thái cuối của nó dùng cho classification, và đã được pretrain nên mang thông tin câu |
| Self-attention nhiều head | 4 head, d_k=32, 2 layer | nhiều head (12) và nhiều layer (12 cho bản base), cùng công thức softmax(QKᵀ/√d_k)V |
| FFN | Linear(128→256)–ReLU–Linear(256→128) | d_ff = 4·d_model, hàm kích hoạt GELU |
| Residual + LayerNorm | post-norm: LN(x+Sublayer(x)) sau attention và sau FFN | cùng vị trí (sau mỗi sublayer); residual giữ gradient qua nhiều layer (demo đồ chơi: chuẩn Jacobian sau 40 lớp 5.7e-12 nếu chỉ F, 58.7 nếu x+F), LayerNorm giữ scale |
| Mask | padding mask | padding mask (+ mask token cho pretraining) |

## 4. Giới hạn còn lại

- **O(T²)**: peak CUDA 2,184 MB ở T=1024 (B=16) so với 532 MB của LSTM; BERT cũng bị giới hạn độ dài (thường 512 token), trong khi 29% review test vẫn dài hơn 256 token.
- **Truncation**: ở T=128, 74.96% review bị cắt (giữ 51.07% token); phán quyết cuối review có thể bị mất (error_analysis W1, W3).
- **Attention ≠ explanation**: token phân cực nhận nhiều attention hơn (5.3× ở layer 2) nhưng cũng có từ trung tính được attend; mô hình sai vẫn "attend đúng từ".
- **Độ tin cậy kết quả**: 3 seed, 1 split, 5 epoch, 1 cấu hình d_model/d_ff/layers; các chênh lệch ≤ 1–2 điểm (heads, pooling, PE) không phân biệt được với nhiễu. Chưa tách riêng đóng góp của từng thành phần kiến trúc trong so sánh LSTM vs Transformer (khác cả pooling lẫn số tham số không-embedding).
- **Negation/contrast, mơ hồ**: demo cho kết quả sai hoặc quá tự tin ở vài câu (xem `error_analysis.md` mục 4); chưa đủ mẫu để kết luận hệ thống.