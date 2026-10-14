# Dự đoán — chép từ bài viết tay đã có

Nguồn: `23001958_Tran_Xuan_Vu_Lab04.pdf`, trang 3. Giữ nội dung dự đoán gốc, không sửa để khớp kết quả. Chưa có bằng chứng xác nhận bài viết tay được tạo trước lần chạy notebook. Các mức Confidence không có trong bản viết tay nên không tự bổ sung.

## Prediction 1 — TF-IDF + Logistic Regression

**Prediction:** Có khả năng hoạt động tốt cho một số bài toán.

**Reason:** TF-IDF biểu diễn mức độ quan trọng của từ; Logistic Regression có thể học trọng số để phân biệt các class.

**Confidence:** Chưa ghi trong bản gốc.

## Prediction 2 — Average Word2Vec so với TF-IDF

**Prediction:** Average Word2Vec không chắc chắn tốt hơn TF-IDF.

**Reason:** Có khả năng phản ánh semantic similarity, nhưng averaging có thể làm mờ những từ quan trọng.

**Confidence:** Chưa ghi trong bản gốc.

## Prediction 3 — Tăng vocabulary size

**Prediction:** Accuracy không chắc chắn tăng khi tăng vocabulary.

**Reason:** Vocabulary lớn hơn có thể tăng feature space, chứa thêm từ hiếm và nhiễu.

**Confidence:** Chưa ghi trong bản gốc.

## Prediction 4 — Bỏ stopwords

**Prediction:** Performance không chắc chắn tăng.

**Reason:** Một số stopword có thể mang thông tin quan trọng.

**Confidence:** Chưa ghi trong bản gốc.

## Prediction 5 — Accuracy và F1

**Prediction:** Có thể cho kết luận khác nhau.

**Reason:** Accuracy đo tỷ lệ dự đoán đúng trên toàn dataset; F1 kết hợp precision và recall, phản ánh khác biệt khi dữ liệu mất cân bằng.

**Confidence:** Chưa ghi trong bản gốc.

## Phạm vi kiểm chứng

Đọc `experiment_report.md` và tự đối chiếu dự đoán với kết quả. Thí nghiệm unigram/bigram thay đổi đồng thời n-gram và cap vocabulary, nên không phải phép thử riêng ảnh hưởng vocabulary. Notebook chưa có ablation có/không stopword; không đánh dấu Prediction 4 đã được kiểm chứng trên IMDb. Dataset thật cân bằng, nên không dùng kết quả IMDb để thay ví dụ mất cân bằng của Bài 4.
