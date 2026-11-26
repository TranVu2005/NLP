# LAB 10 — Reflection (mục 34)

**LAB 08, LAB 09 và LAB 10 đều dùng pretrained Transformer nhưng giải quyết ba bài toán khác nhau như thế nào?**

```
                    Pretrained Transformer  (bert-base-cased)
                                │
                    H = contextual representations  [B, T, 768]
                                │
        ┌───────────────────────┼────────────────────────┐
        ↓                       ↓                        ↓
 [CLS] representation       Token reps               Token reps (context)
   h[CLS] [B, d]            H [B, T, d]               H [B, T, d]
        ↓                       ↓                        ↓
  Linear(d, C)             Linear(d, 9)              Linear(d, 2)
 logits [B, C]            logits [B, T, 9]       start/end logits [B, T]
        ↓                       ↓                        ↓
 Classification                NER                  Start / End
  (1 nhãn / chuỗi)       (1 nhãn BIO / token)    (1 span / (question, context))
        ↓                       ↓                        ↓
     LAB 08                  LAB 09                   LAB 10
```

Cả ba lab dùng cùng một backbone: encoder pretrained biến mỗi token thành một vector phụ thuộc toàn bộ ngữ cảnh. Backbone này **không biết** bài toán nào cả; thứ quyết định bài toán là *chọn biểu diễn nào đưa vào head, head trả ra cái gì, và target là gì*. LAB 08 chỉ lấy `h[CLS]` — một vector tóm tắt cả chuỗi — nên target là một nhãn mỗi chuỗi và loss là cross-entropy trên C lớp. LAB 09 giữ **mọi** vector token: head giống nhau nhưng áp dụng trên từng vị trí, target là chuỗi nhãn BIO, và vì nhãn thuộc về *từ* trong khi mô hình thấy *subword* nên phải căn chỉnh (subword đầu, `−100`). LAB 10 cũng dùng mọi vector token (của context), nhưng head `Linear(d, 2)` cho hai điểm số mỗi vị trí và softmax **theo chiều T**: target không phải một nhãn mà là **hai vị trí** (start, end), nghĩa là đối tượng được dự đoán chính là *chỉ số trong input*. Vì vậy phải đổi nhãn ký tự của SQuAD sang chỉ số token bằng offset mapping, xử lý context dài bằng sliding window, và đánh giá bằng EM/F1 thay vì accuracy hay entity-F1.

Điều rút ra là sự chênh lệch giữa các bài toán nằm ở **phần rất nhỏ** của hệ thống: head QA chỉ có 1,538 tham số trên 107.7 triệu của model, nhưng phần kỹ thuật dữ liệu (alignment, windowing, giải mã span) mới là nơi dễ sai và tốn công nhất. Cùng một backbone dùng được cho cả ba nhờ pretraining đã mã hoá cú pháp/ngữ nghĩa trong biểu diễn ngữ cảnh; fine-tune chỉ dạy cách *đọc* thông tin ấy theo target của từng task. Đó cũng là lý do target quan trọng hơn kiến trúc: đổi *mức nhãn* (chuỗi → token → cặp vị trí) là đổi bài toán.

Giới hạn của LAB 10 dẫn tới bước tiếp theo: extractive QA chỉ tìm đáp án *trong context được cho*, không trả lời được khi đáp án vắng mặt hoặc cần tổng hợp, và ta phải biết trước context. Hệ thống thực tế có hàng nghìn tài liệu, nên cần retrieval (LAB 11) để chọn context, rồi generation (LAB 12) để tạo câu trả lời: classification → token prediction → span prediction → retrieval → RAG.
