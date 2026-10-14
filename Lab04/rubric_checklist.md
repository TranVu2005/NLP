# Checklist nộp LAB 04 theo W4

Nguồn: W4 trang 13–14. Không bảo đảm điểm khi phần hiểu biết cá nhân chưa hoàn thiện.

| Tiêu chí | Điểm | Bằng chứng / file | Phần cần tự hoàn thiện |
| --- | ---: | --- | --- |
| Theory & calculation | 10 | `calculations.md`, PDF viết tay, `test_lab04.py` | Đối chiếu dòng w chưa rõ trong bản scan |
| Prediction | 10 | `prediction.md`, PDF trang 3 | Confidence chưa ghi; thời điểm trước experiment chưa xác minh |
| Data inspection & split | 10 | notebook, `experiment_report.md`, `class_distribution.png` | Tự giải thích imbalance và leakage |
| Baseline implementation | 15 | notebook TF-IDF/LR, `preprocessing.py`, 9 tests | Hiểu TF-IDF và sigmoid/threshold |
| Representation comparison | 15 | 4 model, đủ validation/test metrics, demo word order và n-gram | Tự diễn giải; không quy chênh lệch cho ngữ nghĩa nếu thiếu bằng chứng |
| Quantitative evaluation | 15 | `results.csv`, `run_summary.json`, `confusion_matrix.png` | Tự diễn giải FP/FN, metric từng lớp |
| Error analysis | 10 | `error_analysis.md`, `error_examples.json` | Tự phân tích nguyên nhân 3 lỗi; bảng score chưa phải nguyên nhân |
| Model interpretation | 5 | `model_interpretability.md`, `model_weights.json` | Tự đánh giá ý nghĩa feature trong task |
| Reflection | 5 | `reflection.md` | Điền câu trả lời cá nhân; hiện là khung |
| Individual learning check | 5 | `learning_check.md` | Tự chuẩn bị 6 câu trong W4 |

## 9 file bắt buộc

`README.md`, `calculations.md`, `prediction.md`, `classification.ipynb`, `preprocessing.py`, `results.csv`, `confusion_matrix.png`, `error_analysis.md`, `reflection.md`.

Nộp kèm requirements, runner, tests, các JSON bằng chứng, bài viết tay và khai báo AI để đối chiếu. Không nộp cache/dependencies, `.pdf_inspection` hoặc kernel/runtime cục bộ. Không cần tạo thêm PDF báo cáo mới theo W4.

## Kiểm tra trước khi nộp

- Dùng `experiment_report.md` đối chiếu đủ 5 experiments, đúng label mapping negative=0, positive=1.
- Giữ split và fitting trên train; chọn cấu hình bằng validation F1, không chọn theo test.
- Nếu đổi code, dữ liệu hoặc cấu hình: chạy lại toàn bộ notebook rồi cập nhật báo cáo; không dùng output cũ.
- Dùng `AI_CONTRIBUTION.md` khai báo cả câu trả lời tham khảo trong notebook.
- Hoàn thiện analysis/reflection/learning check bằng lời của mình. File có mặt chưa đồng nghĩa bài đã đạt đủ rubric.
