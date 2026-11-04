# Error analysis — Transformer (2 layer, 4 head, mean-pool, T=128, seed 42) trên test IMDb

> Soạn bởi AI từ output của notebook (mục 14); sinh viên cần đọc lại review gốc và tự xác nhận nhận định.
> 6 mẫu **chọn ngẫu nhiên có seed** (1 TP + 1 TN + 1 ngẫu nhiên đúng; 1 FP + 1 FN + 1 ngẫu nhiên sai), không chọn tay. 6/6 mẫu đều dài ≥ 75 token và 4/6 bị cắt (>128 token) vì 74.96% review test dài hơn 128.

## Tổng quan

- Test: acc 0.8382; sai 1,618/10,000 (FP 709, FN 909). Confusion matrix: `confusion_matrix.png` ([[4291, 709],[909, 4091]]).
- Review ≤ 128 token (n=2,529): sai 11.94%; > 128 token (bị cắt, n=7,471): sai 17.61%. Chênh lệch gợi ý truncation ảnh hưởng, nhưng review dài cũng khó hơn → không tách được hai yếu tố.
- Từ phủ định/đối lập (danh sách đơn giản: not, no, never, but, however, …): 78.6% review có từ phủ định, nên chỉ báo quá thô: có phủ định sai 17.02% (n=7,861), có đối lập sai 17.32% (n=8,193), không có cả hai sai 9.33% (n=718, nhóm nhỏ).
- Attention (4 head, 500 review test): token "phân cực" (300 từ có |hệ số| lớn nhất của Logistic Regression BoW huấn luyện trên train) nhận attention tương đối **3.36×** (layer 1) và **5.32×** (layer 2) so với trung bình đều (=1), còn token còn lại 0.95× và 0.90×. Đây là tương quan trong dữ liệu, **không** chứng minh mô hình *dùng* các token đó để quyết định.

> **Attention ≠ explanation.** Phần "top token được attend" bên dưới chỉ mô tả phân phối trọng số (layer 2, trung bình theo head và theo query), không phải lý do của dự đoán. Muốn nói về nguyên nhân phải can thiệp (xoá/thay token, ablation), việc này chưa làm. Các nhận định về nguyên nhân sai là *giả thuyết* kèm bằng chứng gián tiếp.

## 3 mẫu đúng

| # | idx | nhãn | P(pos) Tf | P(pos) LSTM | token (cắt?) |
|---|---|---|---|---|---|
| C1 | 7811 | pos | 0.739 | 0.951 | 463 (có) |
| C2 | 1939 | neg | 0.131 | 0.069 | 117 (không) |
| C3 | 2911 | neg | 0.004 | 0.009 | 75 (không) |

**C1 (long sequence / truncation).** Review khen miniseries *Fingersmith* ("this mini series deserves to be cherished and praised", "the acting is so great"). Mô hình chỉ thấy 128/463 token; BoW-LR cho P(pos)=0.926 trên 128 token đầu và 0.995 trên toàn văn → tín hiệu tích cực đã nằm sẵn trong phần được thấy, cắt không làm đổi nhãn. Có đoạn đối lập ("yet i passed it away … however during christmas …") và phủ định ("never ends well") nhưng mô hình vẫn đúng, với độ tin cậy thấp hơn LSTM (0.739 so với 0.951). Token phân cực "great" ở phân vị attention 92 (layer 1) và 98 (layer 2); top-8 token được attend (layer 2) gồm cả từ trung tính (local, light, name, imelda, cassidy) → attention cao không đồng nghĩa với từ mang cảm xúc.

**C2 (negation + contrast + domain term).** Review về phim G.I. Joe: "at best a parody", "if the next one is not better this could spell the end", "don't turn it into the care bears". Có 2 từ phủ định, 1 từ đối lập, tên riêng/domain (beachead, cobra, g i joe) nhưng nhãn đúng với P(pos)=0.131. Token phân cực duy nhất theo LR là "soldier" (hệ số +0.95, dương!, phân vị attention 74/94) — từ này khiến BoW nghiêng sang tích cực, vậy mà mô hình vẫn đúng → mô hình không chỉ dựa vào từ đơn lẻ kiểu BoW; top attended có "best", "parody", "attempt". Không đủ để nói mô hình hiểu phủ định.

**C3 (rõ ràng, nhiễu thấp).** "horrible horrible movie … no plot bad acting unfunny … worst movie" — P(pos)=0.004. Cả 6 token phân cực (worst, unfunny, worse, horrible, stupid, bad) đều ở phân vị attention ≥ 85 ở ít nhất một layer, top-8 gồm worst, stupid, bad. Đây là ca dễ: từ vựng cảm xúc tập trung, review ngắn, không bị cắt.

## 3 mẫu sai

| # | idx | nhãn thật | loại lỗi | P(pos) Tf | P(pos) LSTM | token (cắt?) |
|---|---|---|---|---|---|---|
| W1 | 1567 | neg | FP | 0.759 | 0.836 | 320 (có) |
| W2 | 844 | pos | FN | 0.473 | 0.860 | 84 (không) |
| W3 | 3640 | pos | FN | 0.206 | 0.034 | 266 (có) |

**W1 (truncation + domain term + sarcasm/ngôn ngữ văn học).** Review phê bình chuyển thể tiểu thuyết (*The Bonfire of the Vanities*): đoạn đầu dùng từ vựng văn học (satire, polemic, ambrose bierce, spectre, tom wolfe) **không có token phân cực nào** trong 128 token đầu; phán quyết ("totally wasted", "mere caricatures", "fails to come even close") nằm ở phần **bị cắt**. BoW-LR trên toàn văn P(pos)=0.012 nhưng chỉ trên 128 token đầu là 0.098 — vẫn nghiêng âm, nên cắt không phải lời giải thích duy nhất: mô hình còn bị từ vựng văn học/"flop… satire" đẩy sang dương (cả LSTM cũng sai, 0.836). Attention top-8 (uniformly, ambrose, satire, spectre, wolfe, tails) toàn từ miền chuyên biệt. Giả thuyết: truncation + domain term; chưa kiểm chứng bằng can thiệp.

**W2 (contrast, nhãn gần ranh giới, không bị cắt).** Review khen *Crouching Tiger*-like với "incredible special effects", "eye popping cgi" nhưng chứa phần đối lập mang giọng chê/trung tính ("the plot may be hard to follow", "however an intelligent person should be able to keep up", "would put industrial light and magic to shame"). P(pos)=0.473, sát ranh giới 0.5; LSTM đúng (0.860). Token "incredible" (phân cực dương) được attend mạnh (phân vị 96/85), nhưng top-8 còn có "light", "would", "overall" → attention không chuyển thành quyết định đúng. Dạng lỗi: review nhiều mệnh đề mang cảm xúc ngược nhau; mô hình thiếu tự tin.

**W3 (review hỗn hợp / nhiễu nhãn, truncation).** Review phim kinh dị Đức giá rẻ: "the plot is real horrific … the development of plot and characters are great … it could be perfect but unfortunately some things were a little bit disappointing … crappy german court shows … i was very very disappointed", kết bằng "an interesting movie worth to watch" (phần bị cắt). Nhãn pos nhưng 128 token đầu rất tiêu cực: các token phân cực xuất hiện cả hai phía (perfect +1.24, great +0.88 / disappointing −1.45, unfortunately −0.8) và đều được attend mạnh (phân vị 79–100) → attention không giải quyết được mâu thuẫn. BoW-LR cũng sai (0.154 toàn văn, 0.396 trên 128 token đầu), cho thấy nhiều khả năng đây là review khó/nhãn mơ hồ hơn là lỗi riêng của Transformer. Phần cuối bị cắt mang tín hiệu tích cực ("worth to watch") — lý do để nghi truncation góp phần.

## Nhận xét chung về pattern lỗi (từ 6 mẫu + thống kê — chưa suy rộng)

1. **Truncation**: 2/3 mẫu sai bị cắt và phán quyết cuối review nằm ở phần bị cắt (W1, W3); W2 không bị cắt. Tỉ lệ sai review bị cắt cao hơn không bị cắt (17.61% vs 11.94%).
2. **Từ vựng miền chuyên biệt / không có token phân cực** (W1) khiến mô hình không có điểm tựa.
3. **Ý kiến hỗn hợp / contrast** (W2, W3) → xác suất gần ranh giới hoặc sai.
4. **Ứng dụng demo** (notebook mục 13) củng cố điều này: "At first it looked dull, but it turned into a wonderful film." → P(pos)=0.032 (sai, câu đúng nên là tích cực); "It was okay, I guess." → P(pos)=0.0038 (rất tự tin dù mơ hồ); "I did not hate it; the film is not bad." → 0.006. Chưa đủ để kết luận về negation nói chung (9 câu, 1 mô hình, 1 seed).
5. Nhiễu: thẻ `<br />` xuất hiện dưới dạng token "br" ở một số review (C1: 4, W1: 2, W3: 2); chưa đo ảnh hưởng riêng.
