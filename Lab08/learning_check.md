# LAB 08 — Individual Learning Check (mục 27)

1. **Pretraining khác fine-tuning ở đâu?** Pretraining huấn luyện encoder trên kho văn bản rất lớn với mục tiêu tự giám sát (vd. MLM), không gắn với bài toán cụ thể, để học biểu diễn ngôn ngữ chung. Fine-tuning tiếp tục huấn luyện từ trọng số đó (θ_pretrained → θ_task) trên dataset có nhãn nhỏ hơn của một task, kèm head task-specific.

2. **MLM là gì?** Che một phần token đầu vào (`The cat [MASK] fish.`), model dự đoán token bị che từ ngữ cảnh hai phía; loss = cross-entropy `−Σ_{i∈M} log P(y_i|x)` trên các vị trí bị che.

3. **Tại sao cần tokenizer?** Model chỉ nhận số. Tokenizer biến text thành chuỗi `input_ids` theo vocab cố định mà model đã pretrain, thêm special token, cắt/đệm độ dài và sinh `attention_mask`. Phải dùng đúng tokenizer của checkpoint, nếu không ID không khớp embedding đã học.

4. **Subword token giải quyết vấn đề gì?** Từ hiếm/mới/biến thể hình thái không nằm trong vocab word-level (gây OOV/`[UNK]`) được tách thành mảnh nhỏ có trong vocab (`unbelievably → un ##bel ##ie ##va ##bly`) → vocab nhỏ gọn (30k) nhưng phủ gần như mọi từ. Đổi lại: một từ hiếm tốn nhiều token (độ dài tăng) và nghĩa bị chia nhỏ.

5. **[CLS] thường được dùng thế nào?** Token đặc biệt đứng đầu chuỗi; sau các lớp self-attention, vector tại vị trí đó (`last_hidden_state[:,0]`) được dùng làm biểu diễn của cả câu cho classification: `[CLS] → Dropout → Linear → logits → softmax`. Nó "đại diện" cho chuỗi chỉ vì head và encoder được huấn luyện để gom thông tin vào vị trí này (trong DistilBERT không có NSP, nên điều này phụ thuộc fine-tuning).

6. **Attention mask dùng để làm gì?** Đánh dấu token thật (1) và `[PAD]` (0) để self-attention gán trọng số 0 cho padding (score → −∞ trước softmax) và pooling không tính padding; cho phép batch nhiều câu độ dài khác nhau mà kết quả không phụ thuộc vào lượng padding (đã kiểm chứng ở `calculations.md` §3).

7. **Frozen encoder nghĩa là gì?** `requires_grad=False` cho toàn bộ trọng số encoder: không cập nhật khi train, chỉ train head. Encoder đóng vai trò bộ trích đặc trưng (feature extraction).

8. **Fine-tuning thay đổi những tham số nào?** Toàn bộ: embedding, mọi lớp Transformer (attention, FFN, LayerNorm) **và** head mới khởi tạo. (Biến thể khác như chỉ fine-tune vài lớp trên cùng hoặc adapter/LoRA thì chỉ đổi một phần; lab này fine-tune đầy đủ.)

9. **Vì sao pretrained hữu ích khi downstream dataset nhỏ?** Phần lớn việc học biểu diễn (từ vựng, cú pháp, ngữ nghĩa ngữ cảnh) đã xong từ pretraining; downstream chỉ cần học ánh xạ biểu diễn → nhãn với ít tham số/ít dữ liệu hơn. Mức độ giúp thực tế được đo ở Exp 3 (`data_efficiency.png`) — không nên khẳng định chỉ từ lý thuyết.

10. **Pretrained có đảm bảo không bị domain shift không?** Không. Pretraining trên corpus lớn nhưng không bao phủ mọi domain/style; sau fine-tune trên domain A, hiệu năng ở domain B có thể giảm (đo ở Exp 5). Pretrained giảm nhưng không loại bỏ rủi ro này.

11. **Vì sao tăng sequence length làm chi phí attention tăng?** Self-attention tính điểm cho mọi cặp token: ma trận `T×T` mỗi head/layer ⇒ thời gian và bộ nhớ của phần scores tăng `O(T²)` (gấp đôi T → ~4 lần). Phần projection/FFN chỉ `O(T)`, nên tổng chi phí thực tế nằm giữa tuỳ T và d (đo ở Exp 4).

12. **Confidence cao có đảm bảo prediction đúng không?** Không. Confidence = max softmax chỉ là độ chắc chắn của model; model vẫn có thể sai với xác suất rất cao (xem nhóm High-confidence trong `confidence_analysis.png` và `error_analysis.md`). Confidence ≠ correctness; cần calibration/đánh giá riêng.
