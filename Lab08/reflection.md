# Reflection — Tại sao thực tế không bắt đầu bằng train Transformer từ đầu?

> Văn bản do AI soạn từ kết quả của lab này; sinh viên cần đọc lại và viết lại bằng lời của mình (đề mục 24/29 yêu cầu reflection là sản phẩm cá nhân) — xem README.

Sau LAB 07 ta đã tự xây được Transformer, nhưng LAB 08 cho thấy lý do người ta hiếm khi bắt đầu từ trọng số ngẫu nhiên.

**Data.** Train từ đầu phải học cả từ vựng, cú pháp lẫn nhãn từ chính tập nhỏ của bài toán. Trong thí nghiệm, với 10% IMDb (3,189 mẫu), DistilBERT khởi tạo ngẫu nhiên đạt F1 0.7410 còn bản pretrained đạt 0.8386; ở 100% data hai con số là 0.8398 và 0.8820 (trung bình 3 seed). Khoảng cách thu hẹp từ 0.0977 xuống 0.0422 nhưng không biến mất — pretrained tiết kiệm nhãn, không thay thế nhãn.

**Pretrained representation.** Pretraining (MLM trên kho văn bản lớn) đã học sẵn biểu diễn ngữ cảnh. Bằng chứng gián tiếp: chỉ train head 1,538 tham số trên encoder đóng băng đã cho F1 0.8103, trong khi đoán mò là 0.5. Cũng vì vậy, tăng kích thước model một mình không đủ: DistilBERT ngẫu nhiên 66.4M tham số (F1 0.8404) không hơn nhiều Transformer 4.1M của LAB 07 (0.8349).

**Computational cost.** Chi phí pretraining (rất lớn) do người khác trả một lần, ta chỉ trả chi phí fine-tune: 497 s trên một GPU laptop RTX 3050. Giới hạn: thời gian train scratch (477 s) và pretrained (497 s) gần như bằng nhau vì cùng kiến trúc; lợi ích nằm ở data và chất lượng, không phải thời gian mỗi epoch, và chi phí pretraining ta không đo.

**Downstream task và fine-tuning.** Encoder là phần dùng chung; classification head là phần riêng của task. Fine-tuning cập nhật toàn bộ tham số từ θ_pretrained sang θ_task và tốt hơn frozen 0.0679 F1 (0.8782 so với 0.8103), nghĩa là biểu diễn chung chưa khớp hoàn toàn với task.

**Limitation và domain shift.** Pretrained không phải "hiểu mọi thứ". Model fine-tune trên IMDb chỉ đạt accuracy 0.8516 trên Yelp, kém 0.0691 so với model train ngay trên Yelp (0.9207), và lệch hẳn về dự đoán positive (FP 1,204 so với FN 280). Giới hạn độ dài cũng là hạn chế thực tế: tăng `max_len` từ 128 lên 256 nâng F1 từ 0.8782 lên 0.9136 vì lời kết luận thường nằm cuối review. Cuối cùng, confidence cao không đảm bảo đúng: 245 trong 883 lỗi nằm ở nhóm confidence ≥ 0.9.

Do đó thực hành hợp lý là bắt đầu từ pretrained model rồi đo đạc, chứ không bắt đầu từ model ngẫu nhiên.

**Cầu nối sang LAB 09.** Classification chỉ cho một nhãn cho cả văn bản. LAB 09 chuyển sang token classification (NER): thay vì lấy biểu diễn `[CLS]`, ta gán nhãn (PERSON, LOCATION, …) cho từng token từ `last_hidden_state`, vẫn dùng encoder pretrained. Những vấn đề vừa gặp — subword tokenization, truncation, domain shift — sẽ xuất hiện lại ở cấp độ từng token.
