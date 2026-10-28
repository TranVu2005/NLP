# Reflection (LAB 06, mục 28 và 35)

## 1. Attention giải quyết vấn đề gì của recurrent processing?
Mọi vị trí truy cập trực tiếp mọi vị trí khác (đường đi giữa hai token là O(1) thay vì qua T bước), nên thông tin ở xa không phải đi qua chuỗi hidden state và bị nén dần vào một vector h_T. Tính toán cũng song song được trên các vị trí, trong khi RNN phải chạy tuần tự. Trong lab: self-attention classifier train nhanh gần gấp đôi LSTM (7.5 s so với 14.5 s mỗi lần ở lần chạy cuối; số giây phụ thuộc máy và mỗi lần chạy) và ít dao động giữa các seed hơn (std accuracy 0.0037 so với 0.0294).

## 2. Attention tạo ra vấn đề mới nào?
Chi phí O(T²) về thời gian và bộ nhớ: T = 1000 có 10⁶ phần tử, T = 10 000 có 10⁸ phần tử (400 MB float32 chỉ cho một ma trận); thời gian đo có hệ số mũ log-log ≈ 2 khi T ≥ 500. Ngoài ra self-attention không tự có thứ tự nên cần positional information, và attention pattern không phải là lời giải thích cho dự đoán (xem câu 6).

## 3. LAB 05 hỏi "làm thế nào để duy trì thông tin qua sequence?", LAB 06 hỏi "làm thế nào để chọn thông tin liên quan trong sequence?". Hai câu hỏi khác nhau ở đâu?
LAB 05 giải quyết bằng trạng thái truyền theo thời gian: thông tin muốn đến cuối chuỗi phải được giữ lại qua mọi bước. LAB 06 bỏ ràng buộc đó: query so khớp với key của mọi vị trí rồi lấy tổng có trọng số của value, nên model *chọn* thông tin cần dùng thay vì phải *giữ* tất cả. Trong lab, LSTM dùng h_T còn self-attention dùng context vector của mọi vị trí; và trọng số chọn lọc đó phụ thuộc query (B5: cùng câu, đổi query thì top-token đổi từ `slow`/`good` sang `movie`/`beginning`).

## 4. Làm thế nào để thực hiện attention cho toàn bộ sequence một cách hiệu quả?
Tính cho mọi query cùng lúc bằng phép nhân ma trận (QKᵀ, softmax theo hàng, rồi AV) thay vì vòng lặp theo từng query; cách này tận dụng GPU và batch. Giới hạn vẫn là ma trận T × T, nên với T lớn cần các hướng giảm chi phí (attention thưa, cửa sổ cục bộ…). Đây là câu hỏi mở dẫn sang LAB 07.

## 5. Nếu mỗi token có thể attention tới mọi token khác, làm thế nào xây dựng một architecture hoàn chỉnh để xử lý sequence?
Cần ghép thêm các thành phần mà lab này cố ý chưa làm: multi-head (nhiều không gian attention song song), positional encoding (vì self-attention không biết thứ tự), feed-forward, residual connection và layer normalization, rồi xếp nhiều lớp thành Transformer block (LAB 07).

## 6. Attention có chứng minh model đang reasoning không?
Không. Bằng chứng trong lab: (a) BERT dồn khoảng 50–79% attention mass của các token mục tiêu vào `[CLS]`/`[SEP]`/`.`, nên top-3 attention của `cat`, `adopted`, `sleeping` không cho biết quan hệ cú pháp; (b) trong error analysis, token top-1 của attention ở ca 6028 là `composition`, không phải từ cảm xúc, mà dự đoán vẫn đúng và bền; (c) xóa token cho thấy ảnh hưởng không nhất quán với attention (ca 3286 nằm trong dao động của đối chứng). Attention là bằng chứng về phân phối trọng số, không tự động là bằng chứng về lập luận.

## 7. Kết quả LSTM vs Self-attention nói lên điều gì (và không nói lên điều gì)?
Nói lên: trong cấu hình này (100 token đầu, 3 epoch, 3 seed), self-attention classifier cho accuracy 0.8102 ± 0.0037, cao hơn LSTM 0.7643 ± 0.0294 ở cả 3 seed, train nhanh hơn và ổn định hơn. Không nói lên: attention tốt hơn LSTM nói chung, hay chênh lệch là do cơ chế attention. LSTM có thể học chậm hơn trong 3 epoch, và mean pooling gần giống bag-of-words, nên cần ablation (bỏ position, đổi pooling, train lâu hơn) mới kết luận được. Phép so sánh cũng không cho thấy attention nên luôn thay RNN vì chi phí O(T²).

## 8. Điều còn chưa chắc
Vì sao model bỏ sót câu ngắn như "I like this movie." (p_pos = 0.010) và "This movie is not bad at all." (p_pos ≈ 0). Giả thuyết là model train trên review dài (88.6% test dài hơn 100 token) nên câu ngắn nằm ngoài phân phối, nhưng mình chưa kiểm chứng bằng thí nghiệm riêng.
