# Prediction (mục 12) — viết TRƯỚC khi chạy bất kỳ thí nghiệm nào

> Bản nháp do AI (Claude Code) soạn trước khi chạy model/baseline, dựa trên kiến thức từ LAB 03/05/08; sinh viên cần đọc lại và diễn đạt bằng lời của mình (đề mục 28 không cho giao prediction cho AI — xem *AI-assisted* trong `README.md`). Nội dung **không sửa** sau khi có kết quả; đối chiếu với số liệu ở `README.md` mục 6 và `ner_finetuning.ipynb`.

## P1: Pretrained Transformer thay cho classifier đơn giản dựa trên word features thì NER thay đổi thế nào?
Prediction: Entity-level F1 tăng rõ rệt (kỳ vọng baseline ~0.60–0.75, Transformer ~0.90–0.92); tăng chủ yếu ở **recall** của entity hiếm/chưa gặp và ở **ranh giới + loại entity phụ thuộc ngữ cảnh** (cùng một từ là LOC hay ORG hay PER).
Reason: Baseline (TF-IDF/features từng từ + Logistic Regression) chỉ thấy bản thân từ và vài từ lân cận qua feature tách rời; từ chưa thấy ở train chỉ còn dựa vào hình dạng chữ (viết hoa, hậu tố). Pretrained Transformer đã học từ MLM hàng tỉ từ: tên riêng chưa gặp vẫn có biểu diễn tốt, và self-attention cho mỗi token một vector **phụ thuộc toàn câu** nên phân biệt được `Washington` (người/địa danh/đội bóng) theo ngữ cảnh. Baseline gán nhãn từng token độc lập nên còn sinh chuỗi BIO không hợp lệ (vd. `I-ORG` đứng sau `O`), làm entity-level F1 thấp hơn token-level.
Expected evidence: khoảng cách baseline–Transformer ở entity-level lớn hơn ở token accuracy (accuracy cả hai đều cao vì ~83% token là `O`); baseline có precision cao hơn recall trên entity chưa gặp ở test; số lỗi "wrong type" giảm mạnh ở Transformer.

## P2: Entity dài nhiều token dễ hay khó hơn entity một token (`Microsoft` so với `United Nations Security Council`)?
Prediction: Entity dài **khó hơn** ở mức entity-level (F1 giảm khi độ dài tăng, nhất là 3 và 4+ token), dù từng token bên trong có thể đúng.
Reason: Entity-level yêu cầu khớp **tuyệt đối** ranh giới + loại: entity n token chỉ đúng khi cả n nhãn BIO đúng, và một lỗi (thiếu một từ, thừa một từ, đổi `I-ORG`→`I-MISC`) làm cả entity sai. Nếu mỗi token đúng độc lập với xác suất p thì entity n token đúng ≈ pⁿ. Ngoài ra entity dài chứa từ chức năng (`of`, `the`, `and`) viết thường dễ bị gán `O`, và ranh giới mơ hồ (tên + chức danh, tên tổ chức lồng tên địa danh).
Counter-point (kiểm chứng bằng số liệu, không khẳng định trước): entity một token thường là từ viết tắt/tên ngắn rất mơ hồ (`Washington`, `Reuters`, `Hawk`, mã đội thể thao), nên phần lớn lỗi *loại* nằm ở entity 1 token; entity 2 token (họ tên người) có thể dễ nhất vì có mẫu `B-PER I-PER` rất đều. Dự đoán: F1(2 token) ≈ F1(1 token) ≥ F1(3) > F1(4+), không đơn điệu hoàn toàn.

## P3: Entity hiếm trong training data có gây nhiều lỗi hơn không?
Prediction: Có. Entity (surface string) không xuất hiện hoặc xuất hiện ≤ 1 lần trong train có recall/F1 thấp hơn rõ so với entity xuất hiện nhiều lần, ở cả baseline và Transformer, nhưng **Transformer giảm khoảng cách này nhiều hơn baseline**.
Reason: Baseline học trực tiếp "từ này → nhãn này" nên entity chưa gặp mất hết bằng chứng từ vựng, chỉ còn hình dạng chữ. Transformer dùng biểu diễn pretrain (đã thấy tên đó trong corpus lớn) + ngữ cảnh nên vẫn đoán được, nhưng vẫn kém entity thường gặp, đặc biệt `MISC` (ranh giới khái niệm không rõ) và `ORG` thể thao (tên đội viết như địa danh).
Expected evidence: recall của entity "unseen in train" thấp hơn "seen" ở mọi model; chênh lệch của baseline lớn hơn của Transformer; loại lỗi chiếm nhiều trong nhóm rare là *wrong entity type* và *missing entity*.
