# AI assistance statement

**Tool:** OpenAI Codex.

**Purpose:** hỗ trợ code, debugging, preprocessing, visualization, môi trường thực thi và xuất hồ sơ bằng chứng bằng tiếng Việt.

**Generated:** helpers/tests, runner, mã thí nghiệm và xuất dữ liệu; Markdown bằng chứng, bảng metrics, bảng coefficient; script tạo hồ sơ nộp. Các câu trả lời tham khảo trong notebook và `notebook_answers.json` cũng có AI hỗ trợ.

**Modified:** pipeline chia dữ liệu theo nhóm review trùng; fit TF-IDF và Word2Vec trên train; thêm đủ 4 cấu hình, metrics, confusion matrix, demo/OOV, ví dụ đúng/sai và coefficient. Không điều chỉnh score thực nghiệm để cải thiện báo cáo.

**Verified:** output notebook từ lần chạy kernel mới, 9 unit tests, 4 model không skip; đối chiếu JSON/CSV/confusion matrix và file chi tiết. Script hồ sơ kiểm tra code source, execution/output fingerprint và metric trước khi xuất báo cáo.

## Nguồn nội dung cá nhân

`calculations.md` và `prediction.md` được chép từ `23001958_Tran_Xuan_Vu_Lab04.pdf`, không phải đáp án cá nhân mới do AI tạo. Chưa xác minh thời điểm viết tay. Mức confidence chưa có được ghi là thiếu, không dựng lại dự đoán trước experiment.

W4 trang 12 không cho dùng AI cho prediction trước experiment, calculation, interpretation metrics, chọn nguyên nhân error, reflection và individual learning check. Các câu trả lời AI trong notebook chỉ là tham khảo có khai báo. Sinh viên cần tự hoàn thiện các phần cá nhân và xác nhận hiểu code; báo cáo bằng chứng tự động không chứng minh đã thực hiện việc đó.
