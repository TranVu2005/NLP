# LAB 04 — Phân loại văn bản

## Hồ sơ theo W4 trang 13

9 file bắt buộc: `README.md`, `calculations.md`, `prediction.md`, `classification.ipynb`, `preprocessing.py`, `results.csv`, `confusion_matrix.png`, `error_analysis.md`, `reflection.md`.

File hỗ trợ giữ lại: `imdb.zip` (dữ liệu), `requirements.txt` (dependencies), `test_lab04.py` (tests được notebook gọi), `run_notebook.py` (Restart & Run All bằng kernel mới), `W4.pdf` và `23001958_Tran_Xuan_Vu_Lab04.pdf` (đề và bài viết tay nguồn).

Các bảng so sánh, demo phủ định, top features, hình phân bố lớp và văn bản ví dụ đã nằm trong notebook; không cần báo cáo/JSON trùng lặp riêng.

## Dữ liệu và split

50,000 review IMDb; negative/positive đều 25.000 (50%); 418 dòng trùng ngoài lần đầu; missing text/label=0. Archive không ghi upstream/release hoặc official split. SHA-256: `73a235bc5fc4df57bb5d517afa480fe6bfd4e2afc25dc5e5867fc87f2d25614d`.

Seed=42; split theo nhóm review trùng: train=35.000, validation=7.500, test=7.500; overlap giữa các split bằng 0. TF-IDF/Word2Vec chỉ fit trên train. Label mapping negative=0, positive=1; binary metrics dùng positive. Model chọn theo validation F1: `tfidf_logistic_unigram_bigram`.

## Kết quả test đã chạy

| Model | Accuracy | Precision | Recall | F1 | Vocabulary | Fit time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_logistic_unigram | 0.896400000 | 0.885803270 | 0.910133333 | 0.897803499 | 50000 | 12.515986 |
| average_word2vec_logistic | 0.867733333 | 0.864814815 | 0.871733333 | 0.868260292 | 58892 | 611.255613 |
| tfidf_linear_svm_unigram | 0.896000000 | 0.889355008 | 0.904533333 | 0.896879958 | 50000 | 13.328219 |
| tfidf_logistic_unigram_bigram | 0.907066667 | 0.898044329 | 0.918400000 | 0.908108108 | 100000 | 40.423605 |

Baseline confusion matrix: `[[3310,440],[337,3413]]`; hàng true/cột predicted theo negative, positive; FP=440, FN=337. Word2Vec time gồm train embeddings, tạo vector và fit LR. Unigram/bigram khác cả n-gram và vocabulary cap, nên không quy toàn bộ chênh lệch cho bigram.

## Chạy lại

Tại thư mục Lab04:

```powershell
python -m pip install -r requirements.txt
python run_notebook.py
```

Hoặc mở `classification.ipynb`, chọn Restart Kernel & Run All. Runner tạo kernel mới, chỉ lưu notebook chính khi toàn bộ cell thành công. Output thực nghiệm đã có đủ 4 model; 17 code cells đã chạy và 9 tests đạt. Lần tinh gọn này giữ nguyên code/output của notebook, không huấn luyện lại.

Các JSON và hình phụ có thể được tạo lại khi chạy notebook, nhưng không phải deliverables và được `.gitignore` bỏ qua. Markdown số liệu hiện có gắn với output hiện tại; nếu chạy lại, tự đối chiếu và cập nhật thời gian/metric, không giữ Markdown cũ nếu kết quả đổi.

## Phần cá nhân còn thiếu

Calculation/prediction được chép từ PDF viết tay; confidence và thời điểm trước experiment chưa xác minh. `error_analysis.md` chứa bằng chứng 3 đúng/3 sai nhưng còn phần nguyên nhân sinh viên tự viết. `reflection.md` còn là khung. W4 trang 12–13 yêu cầu sinh viên tự hoàn thiện reflection, diễn giải và individual learning check; câu trả lời tham khảo AI trong notebook không thay thế các phần này.

## AI assistance statement

**Tool:** OpenAI Codex.

**Purpose:** hỗ trợ code, debugging, preprocessing, visualization, môi trường thực thi và xuất hồ sơ bằng chứng bằng tiếng Việt.

**Generated:** helpers/tests, runner, mã thí nghiệm và xuất dữ liệu; Markdown bằng chứng, bảng metrics, bảng coefficient; script tạo hồ sơ nộp. Các câu trả lời tham khảo trong notebook và các Markdown câu trả lời trong notebook cũng có AI hỗ trợ.

**Modified:** pipeline chia dữ liệu theo nhóm review trùng; fit TF-IDF và Word2Vec trên train; thêm đủ 4 cấu hình, metrics, confusion matrix, demo/OOV, ví dụ đúng/sai và coefficient. Không điều chỉnh score thực nghiệm để cải thiện báo cáo.

**Verified:** output notebook từ lần chạy kernel mới, 9 unit tests, 4 model không skip; đối chiếu JSON/CSV/confusion matrix và file chi tiết. Script hồ sơ kiểm tra code source, execution/output fingerprint và metric trước khi xuất báo cáo.

## Nguồn nội dung cá nhân

`calculations.md` và `prediction.md` được chép từ `23001958_Tran_Xuan_Vu_Lab04.pdf`, không phải đáp án cá nhân mới do AI tạo. Chưa xác minh thời điểm viết tay. Mức confidence chưa có được ghi là thiếu, không dựng lại dự đoán trước experiment.

W4 trang 12 không cho dùng AI cho prediction trước experiment, calculation, interpretation metrics, chọn nguyên nhân error, reflection và individual learning check. Các câu trả lời AI trong notebook chỉ là tham khảo có khai báo. Sinh viên cần tự hoàn thiện các phần cá nhân và xác nhận hiểu code; báo cáo bằng chứng tự động không chứng minh đã thực hiện việc đó.
