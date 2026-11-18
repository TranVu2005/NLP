# LAB 09 — Calculations

> Mọi con số được tính/chạy thật: BIO và entity-level metric kiểm tra bằng `assert` trong `tokenizer_alignment.ipynb` / `ner_finetuning.ipynb` và `test_ner_core.py`; shape kiểm chứng bằng `bert-base-cased` thật. Văn bản do AI soạn nháp — sinh viên cần đọc lại và tự diễn đạt (đề mục 28 không cho giao BIO exercise, alignment reasoning, metric interpretation cho AI) — xem *AI-assisted* trong `README.md`.

## 1. Calculation Exercise 1 — BIO (mục 6)

`Elon Musk founded SpaceX in California.` với `Elon Musk → PER`, `SpaceX → ORG`, `California → LOC`.

| Elon | Musk | founded | SpaceX | in | California | . |
|---|---|---|---|---|---|---|
| B-PER | I-PER | O | B-ORG | O | B-LOC | O |

7 token, 3 entity, 3 token `O`. `Elon Musk` là entity 2 token: `B-PER` mở entity, `I-PER` tiếp tục; `SpaceX`, `California` là entity 1 token nên chỉ có `B-`. Đọc ngược chuỗi nhãn ra đúng `[(PER,0,2), (ORG,3,4), (LOC,5,6)]` (assert trong notebook).

## 2. Calculation Exercise 2 — Entity boundary (mục 7)

`University of California Berkeley` = một organization ⇒ `University→B-ORG, of→I-ORG, California→I-ORG, Berkeley→I-ORG`.

**Vì sao không gán tất cả token là ORG?** NER phải nhận ra *ranh giới*, không chỉ *loại*:

- Nhãn phẳng `ORG ORG ORG ORG` không phân biệt được 1 entity 4 token với 2 entity (`[University of][California Berkeley]`) hay 4 entity 1 token.
- Hai entity cùng loại đứng liền nhau không tách được nếu không có tín hiệu "bắt đầu": `Chelsea Arsenal` nhãn `B-ORG B-ORG` → 2 entity `(0,1),(1,2)`, còn `B-ORG I-ORG` → 1 entity `(0,2)`, còn `ORG ORG` thì mơ hồ (kiểu dòng kết quả bóng đá trong CoNLL).
- Đánh giá entity-level cần span khớp tuyệt đối; không có `B-/I-` thì không định nghĩa được span.

## 3. Label alignment & IGNORE (mục 8–11)

Câu test id 0 (`SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .`): 12 word → 33 token (gồm `[CLS]`, `[SEP]`), vì `JAPAN → J ##AP ##AN`, `SOCCER → S ##OC ##CE ##R`… (cased WordPiece tách chữ IN HOA thành nhiều mảnh). Với **Strategy A** chỉ subword đầu của mỗi word mang nhãn: 12 token có nhãn, 21 token bị `IGNORE (−100)` (2 special + 19 subword không đầu).

$$L = -\frac{1}{|\{i: y_i \ne IGNORE\}|}\sum_{i:\, y_i \ne IGNORE} \log P(y_i \mid x)$$

Với logits ngẫu nhiên seed 0 trên câu này: loss tính tay (log-softmax, trung bình trên 12 vị trí có nhãn) = **2.5163** = `nc.token_loss`. Thay đổi logits ở vị trí IGNORE không đổi loss (test `test_shapes_and_ignore_index_loss`); thay ở vị trí có nhãn thì đổi.

**IGNORE không có nghĩa token không tồn tại**: che các subword IGNORE khỏi attention làm biểu diễn của 12 token có nhãn lệch trung bình 10.598 (L2, max 12.222) — các subword này vẫn đi qua encoder và cung cấp ngữ cảnh cho token khác; chỉ *nhãn* của chúng không tạo gradient trực tiếp.

**Strategy chọn: A** (nhãn ở subword đầu; dự đoán cũng lấy ở subword đầu). Lý do và đánh đổi: xem `tokenizer_alignment.ipynb` mục 3 và `README.md` mục 2. Trên toàn train: 203,621 word → 300,677 token (gồm 28,082 special) ⇒ **R = 1.339**; Strategy A có 203,621 token có nhãn, bỏ nhãn 68,974 subword không đầu (25.3% token non-special); Strategy B có 272,595 token có nhãn (thêm 68,974 nhãn do tokenizer sinh, trong đó 30,987 là `I-X`).

## 4. Calculation Exercise 3 — Shape (mục 15)

$B=8,\ T=128,\ d=768,\ C=9$:

| Tensor | Shape | Kiểm chứng (bert-base-cased) |
|---|---|---|
| hidden states $H$ | $8 \times 128 \times 768$ | `(8, 128, 768)` |
| $W$, $b$ | $768 \times 9$, $9$ | head = 6,921 tham số = 768·9 + 9 |
| logits $Z = HW + b$ | $8 \times 128 \times 9$ | `(8, 128, 9)` |
| labels $Y$ | $8 \times 128$ | `(8, 128)` |

Một batch có $B \cdot T = 1{,}024$ vector logits ($z_i \in \mathbb{R}^9$); chỉ các vị trí có nhãn ≠ −100 góp vào loss. Tổng tham số model 107,726,601 (encoder 107,719,680 + head 6,921). Khác LAB 08: head nhận *mọi* token ($T$ vector $d$ chiều) thay vì chỉ vector `[CLS]` ($1$ vector).

## 5. Calculation Exercise 4 (tự đặt) — Precision / Recall / F1 entity-level (mục 19–20)

Gold: `New York`=LOC, `Obama`=PER, `Google`=ORG. Prediction: `New`=LOC (thiếu `York`), `Obama`=PER, `Google`=LOC (sai loại), `Monday`=ORG (thừa).

| | Entity | Kết quả |
|---|---|---|
| TP | `Obama` (PER) | 1 |
| FP | `New`[LOC] (sai ranh giới), `Google`[LOC] (sai loại), `Monday`[ORG] (thừa) | 3 |
| FN | `New York`[LOC], `Google`[ORG] | 2 |

$P = \frac{TP}{TP+FP} = \frac{1}{4} = 0.25$, $R = \frac{TP}{TP+FN} = \frac{1}{3} = 0.3333$, $F_1 = \frac{2PR}{P+R} = \frac{2 \cdot 0.25 \cdot 0.3333}{0.5833} = \frac{2}{7} = 0.2857$ (assert trong notebook và test).

**Cùng ví dụ ở token-level**: 8 token, 5 đúng (tính cả `O`) ⇒ accuracy = 0.625 — trong khi entity-F1 = 0.286. Token `New`, `Obama`, … đúng nhưng entity `New York` vẫn sai: token-level đếm *đúng từng phần*, entity-level chỉ đếm khi **cả ranh giới và loại** đúng. Với tập CoNLL, ~83% token là `O` nên accuracy bị thổi phồng (model đoán toàn `O` vẫn có accuracy ≈ 0.83, entity-F1 = 0).

`New York` gold = `B-LOC I-LOC`, prediction `B-LOC O` (ví dụ đề bài): token `New` đúng nhưng entity không nhận diện được ⇒ TP = 0, FP = 1 (`New` như một entity 1 token), FN = 1 — test `test_new_york_example_token_correct_but_entity_wrong`.

## 6. Cách xác định TP/FP/FN ở entity-level

Mỗi entity = bộ ba `(câu, loại, [start, end))`, trích từ chuỗi BIO theo quy ước conlleval (`I-X` mở đầu entity nếu không ở sau `B-X/I-X` cùng loại). TP = |gold ∩ pred|, FP = |pred − gold|, FN = |gold − pred|, micro-average trên toàn test; theo loại: lọc theo `type`; theo độ dài: lọc theo `end − start`. Kết quả cài tay khớp đến 6 chữ số thập phân với `seqeval` trên cả baseline và Transformer (kiểm tra độc lập, xem `README.md` mục 5).
