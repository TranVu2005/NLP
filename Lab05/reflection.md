# 1. Những dự đoán ban đầu và kết quả thực nghiệm

Trước khi thực hiện bài lab, em dự đoán rằng RNN, LSTM và GRU có khả năng xử lý thông tin thứ tự từ tốt hơn các phương pháp biểu diễn văn bản tĩnh. Em cũng cho rằng LSTM có khả năng học những phụ thuộc dài tốt hơn RNN thông thường, nhưng không chắc chắn sẽ đạt kết quả cao hơn trên mọi bộ dữ liệu.

Sau khi thực hiện thí nghiệm trên bộ dữ liệu IMDb gồm 50.000 bài đánh giá, em nhận thấy kết quả giữa các mô hình có sự khác biệt khá rõ ràng.

TF-IDF kết hợp Logistic Regression đạt F1 = 0.8971, trong khi RNN đạt 0.6690, LSTM đạt 0.7690 và GRU đạt 0.8141 trên tập test.

Trong ba mô hình xử lý chuỗi, GRU có kết quả tốt nhất, tiếp theo là LSTM và RNN. Kết quả này phù hợp với dự đoán rằng các kiến trúc có cơ chế điều khiển thông tin có thể hoạt động hiệu quả hơn RNN cơ bản.

Tuy nhiên, TF-IDF + Logistic Regression vẫn đạt F1 cao nhất. Điều này giúp em nhận ra rằng một mô hình có kiến trúc phức tạp hơn không nhất thiết mang lại kết quả tốt hơn, đặc biệt khi thời gian huấn luyện và cấu hình mô hình còn hạn chế.

# 2. Sự khác biệt giữa RNN, LSTM và GRU

Qua bài lab, em hiểu rõ hơn cách các mô hình recurrent xử lý văn bản. Khác với Average Word2Vec, RNN đọc từng token theo thứ tự và cập nhật hidden state, nhờ đó có khả năng lưu trữ thông tin từ các từ đã xuất hiện trước đó.

Tuy nhiên, RNN cơ bản có thể gặp khó khăn khi cần duy trì thông tin qua nhiều timestep. LSTM và GRU được thiết kế với các cơ chế gate để kiểm soát việc lưu giữ và cập nhật thông tin.

Kết quả thực nghiệm cho thấy RNN có F1 thấp nhất là 0.6690, LSTM đạt 0.7690, còn GRU đạt 0.8141. GRU cũng có thời gian huấn luyện khoảng 36.42 giây, thấp hơn LSTM với 38.44 giây, trong khi RNN nhanh nhất với 33.71 giây.

Từ đó, em nhận thấy GRU là một lựa chọn khá cân bằng giữa hiệu năng phân loại và thời gian huấn luyện trong thí nghiệm này.

Dù vậy, kết quả được thực hiện với một random seed và ba epoch nên chưa đủ để khẳng định GRU luôn tốt hơn LSTM hoặc RNN. Em cũng nhận ra rằng khả năng xử lý thứ tự từ của kiến trúc không đồng nghĩa với việc mô hình chắc chắn đã học được mọi quan hệ ngữ cảnh.

# 3. Ảnh hưởng của sequence length và truncation

Một trong những kết quả khiến em chú ý nhất là ảnh hưởng của max_length đến hiệu quả phân loại.

Khi sử dụng GRU với max_length = 50, mô hình chỉ giữ lại khoảng 21.26% tổng số token trong tập train và đạt F1 = 0.7659.

Khi tăng max_length lên 100, tỷ lệ token được giữ lại tăng lên 41.12%, đồng thời F1 tăng lên 0.8141.

Với max_length = 200, mô hình giữ được 68.35% token và đạt F1 = 0.8433.

Kết quả này cho thấy việc giữ lại nhiều ngữ cảnh hơn đã giúp cải thiện hiệu năng trong những cấu hình được thử nghiệm. Tuy nhiên, thời gian huấn luyện cũng tăng từ 23.98 giây lên 65.11 giây khi tăng độ dài từ 50 lên 200.

Em nhận thấy đây là sự đánh đổi giữa lượng thông tin được giữ lại và chi phí tính toán. Nếu sequence quá ngắn, mô hình có thể bỏ mất những từ phủ định, từ thể hiện cảm xúc hoặc phần kết luận của một bài đánh giá dài.

Vì vậy, em cho rằng không nên lựa chọn max_length chỉ dựa vào tốc độ huấn luyện, mà cần xem xét phân phối độ dài văn bản và kết quả trên validation set.

# 4. Phân tích lỗi và giới hạn của sequence models

Khi xem xét learning curves, em nhận thấy RNN có dấu hiệu overfitting sau epoch thứ hai. Validation loss giảm xuống 0.674657 ở epoch 2 nhưng tăng lên 0.689945 ở epoch 3, trong khi training loss vẫn tiếp tục giảm.

Điều này cho thấy mô hình có thể đang học tốt hơn trên training data nhưng chưa cải thiện khả năng tổng quát hóa. Trong khi đó, validation loss của LSTM và các cấu hình GRU vẫn giảm đến epoch thứ ba.

Ở phần error analysis, một ví dụ đáng chú ý là bài đánh giá có nội dung bắt đầu bằng "This movie was so frustrating". Nhãn thực tế là negative nhưng GRU dự đoán positive với xác suất khoảng 0.7680.

Bài đánh giá này có 123 token nhưng chỉ 100 token đầu được đưa vào mô hình. Vì vậy, truncation là một yếu tố cần xem xét, mặc dù chỉ dựa vào trường hợp này chưa thể khẳng định nó là nguyên nhân trực tiếp gây ra lỗi.

Một kết quả khác khiến em bất ngờ là trong thí nghiệm long-range dependency, cả 54/54 trường hợp kiểm thử đều được các mô hình dự đoán là positive, kể cả những câu có từ "terrible". Điều này cho thấy các mô hình chưa phân biệt tốt những câu kiểm thử tổng hợp đó.

Ngoài ra, ở các sequence dài 200 token, những từ quan trọng xuất hiện sau vị trí 100 bị cắt bỏ. Khi thông tin không được đưa vào input thì mô hình không thể sử dụng nó để dự đoán.

Qua đó, em nhận thấy sequence models vẫn có những giới hạn liên quan đến độ dài văn bản, thông tin bị cắt bỏ và khả năng học những quan hệ phụ thuộc phức tạp.

# 5. Bài học rút ra

Sau khi hoàn thành LAB 05, điều quan trọng nhất em hiểu được là việc xử lý văn bản theo thứ tự mang lại nhiều khả năng hơn so với chỉ sử dụng một vector biểu diễn cố định.

RNN có thể cập nhật hidden state qua từng token, trong khi LSTM và GRU bổ sung cơ chế kiểm soát thông tin để cải thiện việc ghi nhớ. Tuy nhiên, kết quả thực nghiệm cũng cho thấy những mô hình này không tự động hoạt động tốt hơn các phương pháp đơn giản như TF-IDF.
