# LAB 04 - Phân loại văn bản

Đã chạy đủ cả 4 cấu hình bằng kernel Jupyter mới. 9/9 unit tests đạt; không có thí nghiệm bị bỏ qua. Markdown và các tệp chi tiết được tổng hợp từ output thật.

Corpus: 50,000 review IMDb từ imdb.zip có sẵn trong workspace; CSV không chứa official split. Archive không ghi upstream hoặc release. SHA-256: `73a235bc5fc4df57bb5d517afa480fe6bfd4e2afc25dc5e5867fc87f2d25614d`.

Seed=42; nhãn negative=0, positive=1; binary metrics dùng positive. Chia theo nhóm văn bản trùng bằng StratifiedGroupKFold 14/3/3 của 20 fold: `{'train': {'samples': 35000, 'class_counts': {'positive': 17500, 'negative': 17500}}, 'validation': {'samples': 7500, 'class_counts': {'positive': 3750, 'negative': 3750}}, 'test': {'samples': 7500, 'class_counts': {'negative': 3750, 'positive': 3750}}}`. Trùng chéo: `{'train_validation': 0, 'train_test': 0, 'validation_test': 0}`. Không downsample.

## Kết quả test

| Mô hình | Accuracy | Precision | Recall | F1 | Vocabulary | Thời gian (s) |
|---|---:|---:|---:|---:|---:|---:|
| tfidf_logistic_unigram | 0.896400000 | 0.885803270 | 0.910133333 | 0.897803499 | 50000 | 12.515986 |
| average_word2vec_logistic | 0.867733333 | 0.864814815 | 0.871733333 | 0.868260292 | 58892 | 611.255613 |
| tfidf_linear_svm_unigram | 0.896000000 | 0.889355008 | 0.904533333 | 0.896879958 | 50000 | 13.328219 |
| tfidf_logistic_unigram_bigram | 0.907066667 | 0.898044329 | 0.918400000 | 0.908108108 | 100000 | 40.423605 |

Cấu hình chọn theo validation F1: `tfidf_logistic_unigram_bigram`. Test chỉ đánh giá những cấu hình đã xác định trước; không chỉnh model theo test.

Confusion matrix baseline test: `[[3310, 440], [337, 3413]]`; hàng thực tế/cột dự đoán theo negative, positive. FP=440, FN=337.

## Cấu hình và giới hạn

TF-IDF dùng min_df=2; unigram có max_features=50.000, unigram+bigram=100.000. So sánh dùng cùng split và Logistic Regression, nhưng khác cap vocabulary nên không quy toàn bộ chênh lệch cho n-gram. Word2Vec Skip-gram: vector_size=100, window=5, min_count=2, epochs=10, workers=1, seed=42; PYTHONHASHSEED=42. Chỉ học từ train; document không có từ đã biết nhận vector 0. Thời gian Word2Vec gồm train embedding, tạo vector và fit LR. Hai câu demo không thay thế đánh giá định lượng trên test.

## Tệp

- classification.ipynb: các phần thí nghiệm, 17 code cells, output và Markdown tiếng Việt.
- preprocessing.py và test_lab04.py: helper và 9 numerical/preprocessing tests.
- results.csv: metrics validation/test, confusion matrices, demo và dataset facts; input_text xác định câu demo.
- class_distribution.png và confusion_matrix.png: biểu đồ thực tế, ảnh trạng thái cũ đã được thay.
- run_summary.json: đầy đủ kết quả, split, phiên bản, matrix, report từng lớp và model selection theo validation.
- word_order_demo.json: TF-IDF, hai vector Word2Vec 100 chiều, OOV và xác suất.
- error_examples.json: 3 đúng/3 sai, full text, nhãn, xác suất và feature hỗ trợ.
- notebook_answers.json: câu trả lời tham khảo do AI hỗ trợ, gắn với SHA-256 của kết quả thực nghiệm để tránh dùng câu trả lời cũ khi kết quả đổi.
- model_weights.json: top-20 hệ số dương/âm, ánh xạ nhãn và term/bigram thật.
- requirements.txt: phiên bản thư viện đã chạy.
- run_notebook.py và annotate_results.py: chạy kernel mới và cập nhật Markdown theo output.

## Chạy lại

Trong môi trường Python phù hợp tại Lab04:

```powershell
python -m pip install -r requirements.txt
python run_notebook.py
```

Trên workspace hiện tại, dùng Python đóng gói tại `C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`; runner tự nạp thư viện trong `.lab01-deps` dùng chung. Notebook báo lỗi nếu thiếu thư viện bắt buộc và assert đủ 4 model trước khi export.

## AI hỗ trợ và kiểm chứng

Tool: OpenAI Codex. Mục đích: code/debug/thực nghiệm/đồ thị và tổng hợp Markdown từ output. Đã xác minh: 9/9 tests, kernel mới chạy toàn bộ notebook, 4 model, CSV và dữ liệu chi tiết. Không viết thay reflection hoặc nguyên nhân lỗi cá nhân. Phiên bản: `{'numpy': '2.5.4', 'pandas': '3.0.6', 'matplotlib': '3.11.2', 'scikit-learn': '1.9.1', 'gensim': '4.4.0', 'nbclient': '0.11.0', 'ipykernel': '7.4.0'}`.

## Hồ sơ theo W4

- calculations.md và prediction.md: chép từ PDF bài viết tay đã có; không tạo dự đoán mới sau khi biết kết quả. Confidence và thời điểm trước experiment chưa được xác minh.
- experiment_report.md: báo cáo 5 experiments, metrics, chênh lệch và giới hạn từ output thật.
- error_analysis.md: văn bản đầy đủ, nhãn, score và contribution cho 3 đúng/3 sai; sinh viên tự viết nguyên nhân trong phần riêng.
- model_interpretability.md: top-20 hệ số dương/âm, label mapping và coefficient bigram.
- reflection.md và learning_check.md: khung cho phần sinh viên tự hoàn thiện theo W4 trang 12–13.
- AI_CONTRIBUTION.md và rubric_checklist.md: khai báo hỗ trợ AI và đối chiếu 100 điểm. Các câu trả lời tham khảo AI trong notebook không được coi là bài cá nhân.
- prepare_submission.py và submission_manifest.json: kiểm tra JSON/CSV, xuất báo cáo và SHA-256; giữ nguyên notebook/code/output.

Sau khi chạy notebook, chạy `python prepare_submission.py`. Runner cũng tự gọi script này khi có file. Các vùng AUTO được cập nhật; phần sinh viên viết bên ngoài vùng AUTO được giữ nguyên. Bài chưa hoàn chỉnh phần cá nhân khi còn dấu … trong error analysis/reflection/learning check.
