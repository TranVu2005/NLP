# Prediction

## P1: Transformer xử lý token song song hơn RNN?
Prediction: Có.
Reason: Self-attention tính cho mọi cặp token bằng phép nhân ma trận cùng lúc; RNN phải đợi h_{t-1}.
Confidence: 90%

## P2: Bỏ positional encoding, Transformer còn biết thứ tự?
Prediction: Không.
Reason: Self-attention (không mask) là permutation-equivariant; embedding không chứa vị trí.
Confidence: 90%

## P3: Cùng tập token, khác thứ tự, không có positional info thì phân biệt được?
Prediction: Không. Output của từng token chỉ bị hoán vị theo thứ tự câu; sau mean pooling hai câu cho cùng vector.
Reason: Mỗi token chỉ nhìn tập token, không nhìn vị trí.
Confidence: 90%

## P4: Hai head có nhất thiết học cùng pattern?
Prediction: Không bắt buộc; thường khác nhưng có thể có head gần như thừa.
Reason: Projection W_i khác nhau và khởi tạo khác nhau, nhưng không có ràng buộc ép đa dạng.
Confidence: 80%

## P5: Tăng số head chắc chắn tăng accuracy?
Prediction: Không.
Reason: d_model cố định nên d_k giảm, số tham số gần như không đổi; có thể bão hòa hoặc dư thừa.
Confidence: 85%

## P6: Transformer luôn nhanh hơn RNN trên mọi sequence length?
Prediction: Không.
Reason: Chi phí O(T²); T ngắn hoặc model nhỏ thì overhead nhỏ, T rất dài thì bộ nhớ attention tăng nhanh.
Confidence: 80%
