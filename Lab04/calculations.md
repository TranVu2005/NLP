# Bài tính toán LAB 04

Nguồn: `23001958_Tran_Xuan_Vu_Lab04.pdf`, trang 1–2. Nội dung dưới được chép từ bài viết tay đã có, chuẩn hóa cách trình bày. PDF gốc được giữ nguyên; chưa xác minh thời điểm làm bài so với lần chạy thực nghiệm.

## Bài 1 — Binary classification

Theo W4: x=[1,2,0], w=[0.5,-0.2,0.4], b=0.1.

```text
z = wᵀx + b
  = 0.5 × 1 + (-0.2) × 2 + 0.4 × 0 + 0.1
  = 0.2

P(y=1|x) = sigmoid(z) = 1 / (1 + exp(-0.2)) ≈ 0.5498
```

Bài viết tay kết luận xác suất class 1 khoảng 54,98%. Dòng khai báo w trong ảnh viết tay có dấu ở đầu chưa rõ; phép thế số phía dưới dùng +0.5 và khớp giá trị w trong W4. Sinh viên cần đối chiếu dòng này với PDF gốc.

## Bài 2 — Prediction threshold

Với P(y=1|x)=0.72:

| Threshold | So sánh | Class dự đoán |
| --- | --- | --- |
| 0.5 | 0.72 ≥ 0.5 | 1 |
| 0.8 | 0.72 < 0.8 | 0 |

Giải thích trong bài viết tay: khi tăng threshold, model yêu cầu xác suất cao hơn để dự đoán positive; có thể tăng precision và giảm recall, ngược lại khi giảm threshold. Giữ cách diễn đạt “có thể”, không coi precision luôn tăng là một bảo đảm.

## Bài 3 — Confusion matrix

Theo bảng đề: TP=80, FN=20, FP=10, TN=90.

```text
Precision = 80 / (80+10) = 8/9
Recall    = 80 / (80+20) = 8/10
F1        = 2 × (8/9) × (8/10) / ((8/9)+(8/10)) ≈ 0.8421
```

Nếu class positive là spam, bài viết tay ghi:

- FP: email bình thường bị phân loại thành spam.
- FN: email spam bị phân loại thành bình thường.

## Bài 4 — Accuracy trap

950 negative, 50 positive; model dự đoán mọi mẫu là negative.

```text
TP=0, TN=950, FP=0, FN=50
Accuracy = 950/1000 = 95%
Recall(positive) = 0/(0+50) = 0
F1(positive) = 0
```

Giải thích trong bài viết tay: model không thật sự tốt; accuracy 95% gây hiểu nhầm vì model hoàn toàn không nhận diện được positive class. Cần xem thêm recall và F1.

## Đối chiếu sau khi làm bài

Các giá trị tính tay trên thuộc bài tập W4, không phải confusion matrix IMDb. Ma trận IMDb thật của baseline là `[[3310,440],[337,3413]]`, thứ tự hàng/cột negative, positive. Notebook và `test_lab04.py` có kiểm chứng sigmoid, threshold và metrics. Không thay bài viết tay bằng số liệu test.
