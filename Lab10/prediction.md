# Prediction (mục 9) — viết TRƯỚC khi chạy bất kỳ thí nghiệm nào

## P1: Question và context chứa nhiều từ giống nhau nhưng chỉ một vị trí thực sự trả lời — model có dễ nhầm không?
Prediction: **Có, nhưng chỉ khi có một câu "mồi" (distractor) giống câu hỏi hơn câu chứa đáp án.** Khi từ trùng nằm *đúng ở câu chứa đáp án* (trường hợp đa số của SQuAD, vì người viết câu hỏi nhìn vào câu đó), từ trùng là tín hiệu tốt: nhóm câu hỏi có overlap từ vựng cao sẽ có F1 **cao hơn** nhóm overlap thấp (kỳ vọng ~88 so với ~70). Khi chèn một câu lấy từ bài khác có overlap với câu hỏi cao nhưng chứa đáp án sai cùng kiểu (cùng là số/ngày/tên riêng), F1 sẽ giảm rõ (kỳ vọng −10 đến −25 điểm F1) và phần lớn lỗi mới là model chọn span *bên trong câu mồi*. Chèn câu không liên quan (overlap thấp) chỉ làm giảm nhẹ (< 2 điểm).
Reason: self-attention giữa token câu hỏi và token context học ra "căn chỉnh" từ vựng; phần lớn model SQuAD dựa mạnh vào việc tìm câu/cụm có từ trùng, rồi chọn thực thể đúng *kiểu* (số/ngày/người) gần đó. Câu mồi cùng kiểu thoả cả hai tín hiệu, nên đánh lừa được model mà không cần model "hiểu" sai.
Expected evidence: tỉ lệ prediction rơi vào câu mồi khi có mồi cùng kiểu cao hơn nhiều so với khi chèn câu ngẫu nhiên; baseline keyword-overlap (chọn câu theo overlap) bị mồi lừa nặng hơn Transformer.

## P2: Answer span càng dài thì boundary prediction có khó hơn không?
Prediction: **Có.** EM giảm gần đơn điệu theo độ dài gold (1 token → 7+ token): kỳ vọng EM ≈ 0.75–0.80 (1 token) → 0.50–0.60 (2–3) → 0.35–0.45 (4–6) → 0.25–0.35 (7+). **F1 giảm chậm hơn EM**, nên khoảng cách F1 − EM tăng theo độ dài (vài điểm ở 1 token, ~20–30 điểm ở 7+ token).
Reason: EM đòi khớp **cả hai** biên; span dài có nhiều cách cắt hợp lý (bỏ/giữ mệnh đề phụ, giới từ, đồng vị ngữ) và hai dự đoán start/end là hai softmax độc lập, nên xác suất khớp cả hai giảm khi mơ hồ biên tăng. Ngược lại overlap token vẫn lớn vì phần lõi của answer thường đúng, do đó F1 vẫn cao. Answer 1 token (thường là số, năm, tên riêng) có biên không mơ hồ.
Counter-point (kiểm chứng bằng số liệu): câu trả lời 7+ token thường là mệnh đề/cụm giải thích, có thể có *ít* gold hơn và chất lượng nhãn kém nhất quán, nên đường cong có thể không hoàn toàn đơn điệu.

## P3: Context chứa nhiều thông tin không liên quan (relevant / irrelevant / distractor / relevant) — performance thay đổi thế nào?
Prediction: Phân biệt **ba loại nhiễu**:
1. *Câu hoàn toàn không liên quan* (ngẫu nhiên từ bài khác): F1 giảm rất ít (< 2 điểm) vì model dễ loại bỏ câu không có overlap với câu hỏi;
2. *Distractor* (overlap từ vựng cao, đáp án sai cùng kiểu): F1 giảm mạnh (−10 đến −25), như P1;
3. *Context dài* (nhiều đoạn nối nhau): F1 giảm khi context vượt `max_length` và bị chia window (đáp án nằm ngoài window hoặc bị cắt biên) — xem Exp 1.
Reason: model được fine-tune trên context SQuAD (một đoạn Wikipedia, ~150 token) nên chưa từng thấy nhiễu có cấu trúc đối kháng; thông tin liên quan nhưng thiếu/đa nghĩa (distractor) mới là điểm yếu, còn "độ dài thuần tuý" thì ảnh hưởng gián tiếp qua truncation.
Expected evidence: bảng F1 theo (không chèn / chèn 2 câu ngẫu nhiên / chèn 1 câu mồi) trên cùng một tập câu hỏi cố định.

## P4: Chỉ đổi thứ tự một số câu trong context nhưng giữ nguyên answer — model có bị ảnh hưởng không?
Prediction: **Ảnh hưởng nhỏ nhưng khác 0.** Chuyển *riêng câu chứa đáp án* lên đầu hoặc xuống cuối context: F1 giảm ≤ 2 điểm (model không có thiên kiến vị trí lớn nhờ fine-tune trên đáp án phân bố khắp context); xáo trộn *toàn bộ* thứ tự các câu: F1 giảm 3–6 điểm vì mất đồng tham chiếu (`He`, `The company`, `this`) và liên kết diễn ngôn giữa câu trả lời và câu trước đó.
Reason: biểu diễn của Transformer dùng position embedding tuyệt đối nhưng đáp án SQuAD không tập trung ở một vị trí cố định; phần lớn câu hỏi chỉ cần *một câu* (self-contained), nhưng một phần nhỏ phụ thuộc câu liền trước (đại từ, "the former").
Expected evidence: Δ F1 của "answer sentence → đầu" ≈ "→ cuối" ≈ 0 ± 2; Δ F1 xáo toàn bộ âm rõ hơn.

## Dự đoán định lượng cho các Experiment (để đối chiếu, không bắt buộc)

- **Exp 1 (context length 128 / 256 / 384, các thành phần khác cố định)**: F1(128) < F1(256) ≲ F1(384); F1(384) − F1(256) < 1 điểm (đa số context SQuAD ≈ 150–200 WordPiece token < 256), còn 128 thấp hơn ~2–5 điểm vì nhiều window không chứa ngữ cảnh cần thiết. **Tăng context length không *luôn* cải thiện**: từ 256 → 384 gần như không đổi; training time của 128 **không** rẻ hơn rõ rệt (nhiều window hơn nên tổng số token xấp xỉ), 384 hơi chậm nhất nếu có nhiều batch padding.
- **Exp 2/3**: Transformer EM ≈ 0.68–0.76, F1 ≈ 0.79–0.85 (train subset ~20k câu hỏi, 2 epoch); baseline keyword-overlap: EM ≈ 0.05–0.15, F1 ≈ 0.15–0.28. Có ≥ 5 ví dụ EM = 0 nhưng F1 > 0 một cách dễ dàng (thừa/thiếu một từ chức năng hoặc modifier, vd. `the city of Paris` so với `Paris`).
- **Exp 4**: câu hỏi overlap cao với câu chứa đáp án có F1 cao hơn câu hỏi paraphrase; chênh khoảng 10–20 điểm F1. Trên bộ cặp paraphrase viết tay (kiểu `capital` vs `government is based`) model vẫn đúng đa số nhưng kém hơn cặp overlap cao.
- **Wrong start vs wrong end**: trong số lỗi mà *chỉ một biên* sai, wrong-end nhiều hơn wrong-start (khoảng 55–60 % so với 40–45 %), vì điểm kết thúc của cụm (giữ hay bỏ giới từ, mệnh đề phụ, hậu tố) mơ hồ hơn điểm bắt đầu (thường là một từ viết hoa/chữ số/danh từ rõ ràng). Lỗi cả hai biên sai (nhầm vị trí) chiếm phần lớn nhất trong tổng số lỗi EM = 0 nhưng chủ yếu do distractor/đa đáp án hợp lý.
- **Baseline vs Transformer**: baseline chọn *đúng câu* (answer nằm trong câu được chọn) khoảng 55–70 % nhưng chọn đúng *span* kém (EM thấp) → khoảng cách chủ yếu ở bước định vị span trong câu, không phải ở bước tìm câu.
