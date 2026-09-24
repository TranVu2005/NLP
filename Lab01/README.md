# LAB 01 — From Text Processing to Search

## Các tệp

- `W1.pdf`: đề bài.
- `c4-train.00000-of-01024-30K.json.gz`: corpus JSONL gzip gồm 30.000 record.
- `implementation.py`: các hàm TF-IDF, sparse search, metrics và tiện ích đọc corpus.
- `test_implementation.py`: unit tests.
- `experiments.ipynb`: thí nghiệm trên corpus và sinh kết quả.
- `results.csv`: kết quả xếp hạng được tạo khi chạy notebook.

## Cách chạy

Mở terminal tại thư mục này, chạy `python -m unittest -v test_implementation`, sau đó chạy toàn bộ cell trong `experiments.ipynb` theo thứ tự từ trên xuống. Notebook ghi `results.csv` vào thư mục hiện tại. Môi trường cần các gói NumPy, SciPy, scikit-learn, pandas, `tokenizers` và Jupyter/IPython đang có sẵn.

## Giả định để tái lập

Đề không chỉ định tokenizer, quy tắc xử lý dấu câu, danh sách stopword, mô hình subword hay nhãn relevance. Pipeline A chuyển chữ thường rồi tách theo khoảng trắng, giữ dấu câu gắn với từ. Pipeline B thay dấu câu/ký tự điều khiển Unicode bằng khoảng trắng, tách theo khoảng trắng rồi bỏ stopword tiếng Anh có sẵn trong scikit-learn. Pipeline C chuẩn hóa như B rồi dùng WordPiece được train cục bộ (vocabulary mục tiêu 20.000, tần suất tối thiểu 2), không tải model pretrained. OOV được tính trên cùng tập query cố định cho cả ba pipeline: tỷ lệ token sau preprocessing không có trong vocabulary đã fit trên corpus. Nhãn relevance trong notebook là nhãn thủ công cho thí nghiệm, không phải nhãn của giảng viên.

Core implementation trên corpus nhỏ dùng IDF không smoothing `log(N / df)` và vocabulary sắp theo alphabet. Thí nghiệm corpus dùng sparse matrix; query được đổi vector bằng vocabulary và IDF đã fit trên corpus.

## Khai báo sử dụng AI

AI đã hỗ trợ:
- Viết và hoàn thiện các hàm TF, IDF, TF-IDF, cosine similarity, sparse indexing, search và metrics đánh giá.
- Tìm lỗi và sửa quy trình chạy thí nghiệm trên corpus.
- Xây dựng phần preprocessing, tìm kiếm, đánh giá và xuất kết quả.
- Các số liệu báo cáo được sinh bằng cách chạy notebook trên corpus được cung cấp. Không có nhãn relevance của giảng viên; notebook ghi rõ tập đánh giá là tập nhãn thủ công.
