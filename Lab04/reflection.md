# Reflection — LAB 04: Text Classification

## 1. Dự đoán nào được kết quả hỗ trợ hoặc chưa thể kiểm chứng?

Trước khi thực hiện thí nghiệm, em dự đoán rằng TF-IDF kết hợp với Logistic Regression sẽ là một mô hình baseline phù hợp cho bài toán phân loại văn bản. Em cũng cho rằng Average Word2Vec không chắc chắn sẽ hoạt động tốt hơn TF-IDF.

Theo kết quả trên test set, TF-IDF + Logistic Regression đạt F1 = **0.897803499**, trong khi Average Word2Vec + Logistic Regression đạt F1 = **0.868260292**. Chênh lệch F1 theo chiều Word2Vec trừ TF-IDF là **−0.029543206**, tương đương khoảng 2.95 điểm phần trăm.

Kết quả này phù hợp với dự đoán ban đầu rằng TF-IDF có thể hoạt động tốt và Average Word2Vec không phải lúc nào cũng tốt hơn. Tuy nhiên, đối với các dự đoán liên quan đến việc tăng vocabulary size hoặc loại bỏ stopwords, em chưa có đủ kết quả thực nghiệm riêng để kiểm chứng.

Ngoài ra, em cũng dự đoán Accuracy và F1 có thể đưa ra những đánh giá khác nhau, đặc biệt khi dữ liệu mất cân bằng. Đây là nhận định hợp lý về mặt lý thuyết, nhưng em chưa có đủ số liệu so sánh Accuracy và F1 để kết luận điều đó đã xảy ra trong thí nghiệm này.

## 2. Representation ảnh hưởng đến kết quả như thế nào?

Qua bài lab, em nhận thấy cách biểu diễn văn bản có ảnh hưởng đáng kể đến kết quả phân loại, ngay cả khi sử dụng cùng thuật toán Logistic Regression.

TF-IDF giữ lại thông tin về những từ xuất hiện trong văn bản, giúp mô hình nhận biết các từ có khả năng phân biệt nhãn. Trong khi đó, Average Word2Vec biểu diễn văn bản bằng cách lấy trung bình vector của các từ, có thể phản ánh một phần thông tin ngữ nghĩa nhưng không trực tiếp mô hình hóa thứ tự từ.

Trong thí nghiệm, TF-IDF đạt F1 khoảng **0.8978**, cao hơn Average Word2Vec với F1 khoảng **0.8683**. Điều này cho thấy TF-IDF phù hợp hơn trong thiết lập thực nghiệm hiện tại, nhưng chưa đủ để khẳng định nó luôn tốt hơn Word2Vec trên mọi bộ dữ liệu.

Một kết quả đáng chú ý khác xuất hiện khi thử hai câu "this movie is good" và "this movie is not good". TF-IDF dự đoán cả hai câu là positive, với xác suất lần lượt là **0.8040** và **0.5100**. Trong khi đó, Average Word2Vec dự đoán câu đầu positive và câu thứ hai negative, với xác suất positive tương ứng là **0.7431** và **0.2492**.

Cả hai câu đều không có từ OOV. Mặc dù cosine similarity giữa hai vector Average Word2Vec khá cao, khoảng **0.9767**, mô hình vẫn đưa ra hai nhãn khác nhau.

Qua ví dụ này, em nhận thấy Average Word2Vec vẫn có thể phân biệt được một số câu phủ định trong thực tế. Vì vậy, không thể kết luận rằng phương pháp này luôn xử lý phủ định kém. Hiệu quả của representation cần được đánh giá dựa trên kết quả cụ thể thay vì chỉ dựa vào đặc điểm lý thuyết.

## 3. Lỗi quan trọng nhất và bằng chứng

Một trường hợp dự đoán sai đáng chú ý trong notebook là văn bản có nhãn thực tế **positive** nhưng mô hình lại dự đoán **negative**.

Đoạn văn bản có nội dung: "This was part of the frustration … no avail. Then there's the satisfaction …".

Mô hình đưa ra xác suất negative là **0.610842833** và xác suất positive là **0.389157167**.

Các feature có mức đóng góp lớn được ghi nhận gồm "no" với giá trị **0.503534**, "cheap" với **0.286379** và "sorry" với **0.262643**.

Điểm đáng chú ý là đoạn văn bản có cả những nội dung mang sắc thái tiêu cực như "frustration", "no avail" và nội dung tích cực như "satisfaction". Vì vậy, đây có thể là một trường hợp mô hình gặp khó khăn khi xử lý văn bản chứa nhiều sắc thái cảm xúc khác nhau.

Các feature được ghi nhận cho thấy một số từ có ảnh hưởng đáng kể đến dự đoán. Tuy nhiên, chỉ từ các giá trị đóng góp này chưa thể khẳng định chính xác nguyên nhân gây lỗi nếu không xem thêm chiều tác động của từng feature và toàn bộ ngữ cảnh văn bản.

Qua trường hợp này, em nhận thấy việc chỉ xem Accuracy hoặc F1 là chưa đủ. Phân tích từng dự đoán sai giúp phát hiện những tình huống mà mô hình có thể chưa xử lý tốt, từ đó đề xuất những thí nghiệm cải thiện phù hợp hơn.

## 4. Giới hạn của mô hình và hướng cải thiện

Một hạn chế đáng chú ý của các phương pháp biểu diễn văn bản đơn giản là khả năng xử lý ngữ cảnh và những cụm từ mang ý nghĩa đặc biệt, chẳng hạn như phủ định.

Trong thí nghiệm với "this movie is good" và "this movie is not good", TF-IDF dự đoán cả hai câu là positive. Tuy nhiên, xác suất positive của câu có từ "not" giảm từ **0.8040 xuống khoảng 0.5100**, cho thấy sự xuất hiện của từ phủ định đã ảnh hưởng đến dự đoán, nhưng chưa đủ để làm thay đổi nhãn tại threshold 0.5.

Average Word2Vec lại dự đoán đúng chiều sắc thái mong đợi ở ví dụ này, mặc dù hai vector có cosine similarity khoảng **0.9767**. Điều đó cho thấy vector gần nhau không nhất thiết dẫn đến cùng một nhãn phân loại.

Nếu được cải thiện mô hình, em muốn thử sử dụng TF-IDF kết hợp unigram và bigram thay vì chỉ sử dụng các từ riêng lẻ. Khi đó, những cụm như "not good" có thể trở thành feature riêng, giúp mô hình học được thêm thông tin về sự kết hợp giữa các từ.

Em sẽ đánh giá thay đổi này bằng cách so sánh F1 giữa hai cấu hình trên cùng validation set, đồng thời kiểm tra các trường hợp chứa từ phủ định. Test set sẽ được giữ độc lập và chỉ dùng để đánh giá cuối cùng, tránh tình trạng data leakage hoặc điều chỉnh mô hình dựa trên dữ liệu test.

## 5. AI đã hỗ trợ những gì và em đã kiểm tra như thế nào?

Trong quá trình thực hiện LAB 04, công cụ AI **Codex** đã được sử dụng để hỗ trợ viết, sửa code Python và thực hiện các thí nghiệm trong notebook.

AI hỗ trợ xây dựng các bước xử lý dữ liệu, triển khai mô hình phân loại, tính toán các metrics và tổ chức kết quả thực nghiệm. Điều này giúp quá trình lập trình và kiểm tra chương trình thuận tiện hơn.

Theo thông tin thực thi được ghi nhận, notebook đã có **9 tests đạt yêu cầu** và **17 code cells được chạy**. Đây là bằng chứng về việc chương trình đã được thực thi và kiểm thử trong môi trường của agent, nhưng không đồng nghĩa với việc em đã tự kiểm tra độc lập tất cả kết quả.

Các số liệu F1, kết quả dự đoán của hai câu demo và trường hợp phân loại sai được sử dụng để hỗ trợ việc đối chiếu kết quả thực nghiệm.

Qua bài lab, em nhận thấy AI có thể hỗ trợ hiệu quả trong quá trình lập trình, nhưng việc hiểu ý nghĩa của các metrics, nhận diện giới hạn của representation và đánh giá nguyên nhân mô hình dự đoán sai vẫn cần được người thực hiện kiểm tra và suy luận cẩn thận.
