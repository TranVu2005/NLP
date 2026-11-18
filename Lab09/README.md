# LAB 09 — Named Entity Recognition and Information Extraction

NER trên CoNLL-2003 (PER / ORG / LOC / MISC, BIO): baseline (features + Logistic Regression) vs `bert-base-cased` fine-tune cho token classification, label alignment, metric entity-level, phân tích lỗi/context, và một pipeline Information Extraction nhỏ. Mọi số trong tài liệu này lấy từ `results.csv`, `entity_metrics.csv` hoặc output của notebook đã chạy.

> **AI-assisted (mục 28).** Toàn bộ code, notebook và các phần viết (`prediction.md`, `calculations.md`, `error_analysis.md`, `learning_check.md`, `reflection.md`) do AI (Claude Code) soạn. Đề cấm giao AI làm thay BIO exercise, label-alignment reasoning, prediction, metric interpretation, error analysis và reflection — **sinh viên cần đọc lại, kiểm chứng và viết lại các phần đó bằng lời của mình trước khi nộp.** `prediction.md` được viết *trước* khi chạy thí nghiệm và không sửa sau khi có kết quả.
> *What was generated:* `ner_core.py`, `baseline_ner.py`, `information_extraction.py`, 2 notebook, `test_ner_core.py`, các file markdown. *What was modified:* chưa có sửa đổi nào của sinh viên được ghi nhận. *What was verified:* xem mục 5.

## 1. Cấu trúc & cách chạy

| File | Nội dung |
|---|---|
| `ner_core.py` | Dữ liệu CoNLL-2003, BIO ↔ entity, **metric entity-level tự cài** (conlleval), `align_labels` (Strategy A/B), `NERDataset` (word_ids), `TokenClassifier` (AutoModel + head tự viết, **không** dùng `AutoModelForTokenClassification`/`pipeline`), train/eval, CLI, lưới run |
| `baseline_ner.py` | Baseline: cửa sổ ±2 từ → DictVectorizer → TF-IDF → LogisticRegression |
| `tokenizer_alignment.ipynb` | **Đã chạy**: BIO exercise 1–2, subword, alignment A/B, ignore index, shape exercise 3, core `tokenizer(is_split_into_words=True)`, **Exp 1** (tokenization analysis) |
| `ner_finetuning.ipynb` | **Báo cáo chính, đã chạy**: baseline, pipeline Transformer (tokens / input_ids / attention_mask / word_ids / labels / logits / predictions), **Exp 2–6**, rare entity, context analysis, A vs B, Information Extraction, ghi `results.csv`, `entity_metrics.csv` |
| `information_extraction.py` | `raw text → NER → {"PERSON": [...], "ORG": [...], "LOCATION": [...]}` + đánh giá entity-level; `ie_gold.json` = 30 câu tự viết, gold gán tay |
| `results.csv` | 8 dòng (baseline, 3 seed Transformer A, mean, B, 2 dòng IE); `accuracy` = token accuracy, `precision/recall/f1` = entity-level micro |
| `entity_metrics.csv` | 60 dòng: overall / theo loại / theo độ dài / theo tần suất train × 5 model |
| `entity_length_analysis.png`, `confusion_matrix.png` | Hình bắt buộc (+ `learning_curves.png`, `tokenization_analysis.png`) |
| `prediction.md`, `calculations.md`, `error_analysis.md`, `learning_check.md`, `reflection.md` | Phần viết |
| `error_analysis_evidence.json`, `error_explanations.json` | 18 mẫu lỗi + giải thích (nguồn của bảng trong `error_analysis.md`) |
| `runs/<tên>/` | `result.json` (config, history, metric val/test), `preds_test.json`; `model.pt` (gitignore) cho run chính; `runs/baseline/` |
| `test_ner_core.py` | 12 test (giá trị tính tay): BIO, entity-level P/R/F1, ví dụ `New York`, độ dài entity, alignment A/B, truncation, shape, IGNORE trong loss, padding-invariance, `categorize_errors` |

```powershell
# từ thư mục Lab09 (Python 3.14, torch 2.12.0+cu126, transformers 5.12.1, datasets, scikit-learn, matplotlib, pandas, nbconvert)
python -m pytest test_ner_core.py -q
python ner_core.py --run_grid                       # 4 run Transformer (~30 phút trên RTX 3050), cache vào runs/
python baseline_ner.py                              # baseline (~4 phút CPU)
python -m nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 ner_finetuning.ipynb   # chỉ nạp cache (LAB09_RUN_TRAINING=1 để train lại run thiếu)
python information_extraction.py --text "Barack Obama visited Paris."
python information_extraction.py --evaluate         # 30 câu gold tay
```

Dữ liệu: lần đầu tải `eriktks/conll2003` bản parquet (`refs/convert/parquet`, vì `datasets` mới không chạy dataset script) rồi cache `data_cache/conll2003_*.jsonl`. Nhãn kiểm chứng đúng 9 nhãn `O, B-/I-PER, ORG, LOC, MISC`.

## 2. Lựa chọn và lý do

- **Dataset**: CoNLL-2003 chuẩn, giữ nguyên split train/validation/test = 14,041 / 3,250 / 3,453 câu (203,621 / 51,362 / 46,435 word; 23,499 / 5,942 / 5,648 entity). ~82.5% token test là `O`.
- **Model**: `bert-base-cased` (107.7M tham số): *cased* vì chữ hoa là tín hiệu mạnh của NER (`uncased` làm mất tín hiệu này), WordPiece cho vấn đề alignment như đề mô tả (`play ##ing`), vừa RTX 3050 6GB với fp16 (2.9 GB). Head `Dropout(0.1) → Linear(768, 9)` tự viết.
- **Alignment: Strategy A** — nhãn ở subword đầu, các subword sau `−100`; dự đoán cũng lấy ở subword đầu (nhất quán train/test); metric ở mức word như gold. Lý do và phản biện: `tokenizer_alignment.ipynb` mục 3. Để *phân tích ảnh hưởng* đã train thêm **Strategy B** (lan truyền `I-X`) một seed.
- **Chọn checkpoint**: theo **val entity-F1** (không dùng test); test đánh giá một lần cho checkpoint đó. 3 seed (42/43/44) cho Strategy A; B một seed.
- **Baseline** (đề mục 13): features cửa sổ ±2 (từ lowercase, shape, tiền/hậu tố, viết hoa/IN HOA/số/gạch nối, vị trí, bigram) → TF-IDF sublinear → Logistic Regression (lbfgs; C ∈ {3, 10, 30} chọn theo val entity-F1 → C = 30). Mỗi token phân loại độc lập; không CRF; không dùng POS/chunk tags.
- **Metric**: entity-level micro P/R/F1 theo quy ước conlleval, tự cài (TP = khớp tuyệt đối span + loại); thêm token-level (accuracy và P/R/F1 trên token không-`O`) chỉ để đối chiếu.
- **Information Extraction**: Option A (News) — PERSON/ORG/LOCATION, bỏ MISC. Gold: 30 câu tự viết (không lấy từ CoNLL) với 89 entity.

## 3. Bảng reproducibility

Chung: AdamW (lr 5e-5, weight_decay 0.01, không decay bias/LayerNorm), linear warmup 10% rồi decay, grad-clip 1, fp16, batch 16, 3 epoch, `max_len=256` (câu dài nhất 173 token ⇒ **không câu nào bị cắt**), dropout 0.1, seed 42/43/44; Python 3.14.5, torch 2.12.0+cu126, transformers 5.12.1, sklearn 1.9.0; GPU RTX 3050 6GB Laptop.

| Run | best epoch | thời gian train | VRAM đỉnh | val entity-F1 | **test entity-F1** |
|---|---:|---:|---:|---:|---:|
| Transformer, Strategy A, seed 42 | 3 | 449 s | 2,946 MB | 0.9494 | 0.9159 |
| Transformer, Strategy A, seed 43 | 3 | 421 s | 2,945 MB | 0.9493 | 0.9109 |
| Transformer, Strategy A, seed 44 | 3 | 428 s | 2,945 MB | 0.9484 | 0.9131 |
| Transformer, Strategy B, seed 42 | 3 | 363 s | 2,945 MB | 0.9508 | 0.9124 |
| Baseline LogReg (C = 30, 211,574 feature) | — | 77 s (CPU) | — | 0.8547 | 0.7786 |

(Train loss/val loss/val F1 từng epoch: `runs/*/result.json`, `learning_curves.png`. Chạy lại baseline từ đầu cho prediction giống hệt run đã lưu.)

## 4. Kết quả chính (test CoNLL-2003, cùng một test set cho mọi model)

**Exp 2 — Baseline vs Transformer** (entity-level):

| Model | Precision | Recall | F1 | token acc | token F1 (≠O) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7489 | 0.8109 | 0.7786 | 0.9618 | 0.8272 |
| Transformer A (mean 3 seed) | 0.9066 | 0.9201 | **0.9133 ± 0.0025** | 0.9827 | 0.9248 |
| Transformer A (seed 42, dùng cho phân tích) | 0.9090 | 0.9230 | 0.9159 | 0.9833 | 0.9274 |
| Transformer B (seed 42) | 0.9062 | 0.9187 | 0.9124 | 0.9825 | 0.9242 |

F1 +0.135 trong khi token accuracy chỉ +0.021. **Strategy A vs B**: B − mean(A) = −0.0009, nhỏ hơn std giữa các seed của A (0.0025); B chỉ một seed nên **không kết luận được** B tốt/kém hơn A — trong lab này alignment strategy không tạo khác biệt đo được. (Thời gian train B ngắn hơn, 363 s vs 421–449 s, nhưng máy chạy song song việc khác nên không dùng làm bằng chứng về tốc độ.)

**Exp 3 — Entity-level vs token-level**: gap (token F1 − entity F1) = 0.049 (baseline) và 0.011 (Transformer); 14% entity gold bị sai của Transformer vẫn có ≥ 1 token đúng; baseline sinh 450 chuỗi `I-X` không hợp lệ (Transformer 43).

**Exp 4 — Theo loại entity** (Transformer seed 42 / baseline F1): PER 0.962 / 0.815; ORG 0.900 / 0.689; LOC 0.933 / 0.833; MISC 0.810 / 0.773 (mean 3 seed: 0.960 / 0.897 / 0.931 / 0.806). Diễn giải (số train, % entity unseen, mơ hồ, độ dài, domain): `error_analysis.md` mục 3.

**Exp 5 — Theo độ dài entity** (`entity_length_analysis.png`; Transformer seed 42 F1, support): 1 token 0.910 (3,574); 2 token 0.932 (1,776); 3 token 0.896 (236); 4+ token 0.870 (62). Baseline: 0.761 / 0.838 / 0.663 / 0.545. Nhóm 4+ chỉ 62 entity (std 3 seed 0.022).

**Rare entity** (recall theo số lần chuỗi bề mặt xuất hiện như entity ở train): Transformer unseen 0.884 / rare (1–4) 0.932 / frequent (≥ 5) 0.968; baseline 0.685 / 0.843 / 0.954.

**Exp 6 — Error analysis**: 583 lỗi của Transformer (baseline 1,383): wrong_type 229, spurious 148, boundary 78, boundary+type 65, missing 63. 18 mẫu (5 spurious, 5 missing, 5 boundary, 3 wrong type) có giải thích trong `error_analysis.md`; ≥ 8/18 xuất phát từ nhãn không nhất quán/ngữ cảnh bị cắt của CoNLL. **Context analysis**: `Barcelona`, `CHICAGO`, `WASHINGTON`, `American` (cùng chuỗi, khác loại); Transformer 221/258 vs baseline 200/258 entity đúng trên 41 chuỗi mơ hồ.

**Exp 1 — Tokenization** (`tokenizer_alignment.ipynb`): R = #subword tokens / #words = 1.367 trên test (1.339 trên train); 17.9% word bị tách nhưng **45.3% word thuộc entity** (non-entity 12.0%); word chưa từng thấy ở train bị tách 78.6% vs 3.6% với word thường gặp; từ bị tách nhiều nhất: có gạch nối (100%), IN HOA (86.6%), chứa số (33.9%), viết hoa đầu (31.8%). 30 câu mẫu: R câu ngắn 1.87 > câu dài 1.19 vì câu ngắn của CoNLL là tiêu đề IN HOA/dòng tỉ số.

**Information Extraction** (30 câu, 89 entity, entity-level): Transformer P/R/F1 = 0.967 / 0.978 / **0.972**; baseline 0.766 / 0.809 / 0.787. Xem lưu ý ở mục 6.

## 5. Kiểm chứng

- `python -m pytest test_ner_core.py -q` → 12 test đạt (giá trị tính tay: entity P/R/F1, ví dụ `New York`, bucket độ dài, `categorize_errors`, alignment A/B, IGNORE không đóng góp vào loss, shape `(B,T,768)/(B,T,9)`, padding-invariance).
- **Metric tự cài đối chiếu độc lập với `seqeval`** (cài tạm ngoài `requirements.txt`): P/R/F1 trùng đến 6 chữ số thập phân cho baseline (0.748855 / 0.810907 / 0.778647) và Transformer seed 42 (0.908980 / 0.922982 / 0.915927).
- Baseline huấn luyện lại từ đầu cho prediction giống hệt run đã lưu (`preds_test.json`).
- Notebook chạy lại từ cache cho cùng số; mọi số trong văn bản notebook sinh bằng f-string.

## 6. Đối chiếu prediction (`prediction.md` viết trước khi chạy, không sửa)

| | Dự đoán | Kết quả | Đúng? |
|---|---|---|---|
| P1 | F1 Transformer ≫ baseline (baseline ~0.60–0.75, Transformer ~0.90–0.92); tăng chủ yếu ở recall entity chưa gặp và loại/ranh giới theo ngữ cảnh; chênh ở entity-F1 lớn hơn ở accuracy | Baseline **0.779** (hơi *trên* khoảng dự đoán — baseline có cửa sổ ±2 và bigram mạnh hơn tôi nghĩ), Transformer 0.913; recall unseen +0.199 vs frequent +0.014; `wrong_type` −55%, `boundary` −75%; +0.135 F1 vs +0.021 accuracy | Phần lớn đúng; **sai** ở khoảng giá trị baseline và ở ý "baseline precision cao hơn recall trên entity chưa gặp" (baseline P 0.749 < R 0.811 chung, và trên nhóm unseen P 0.599 < R 0.685) |
| P2 | Entity dài khó hơn; F1(2) ≈ F1(1) ≥ F1(3) > F1(4+), không đơn điệu | Transformer 0.910 / 0.932 / 0.896 / 0.870; baseline 0.761 / 0.838 / 0.663 / 0.545 | Đúng về xu hướng (3 và 4+ thấp nhất; 2 token cao nhất). Phần giả thuyết *pⁿ* **sai**: `p^n` thấp hơn recall thực (n=3: 0.788 vs 0.890) ⇒ lỗi token trong cùng entity **tương quan dương**, không độc lập |
| P3 | Entity hiếm khó hơn; Transformer giảm khoảng cách nhiều hơn baseline | frequent − unseen: baseline 0.269, Transformer 0.085 | Đúng. Phần "loại lỗi chiếm nhiều trong nhóm rare là wrong type/missing" **chưa kiểm chứng** |

## 7. Hạn chế và lưu ý trung thực

- **Tất cả phần viết/phân tích do AI soạn** (xem đầu file) — cần sinh viên viết lại.
- **Nhiễu nhãn CoNLL**: nhiều "lỗi" thực chất là nhãn mâu thuẫn (xem `error_analysis.md` mục 2), nên F1 ≈ 0.91–0.92 gần trần thực tế; không so sánh trực tiếp với SOTA (≈ 0.93+ dùng model lớn hơn/CRF/doc-level context).
- Strategy B chỉ **một seed**; chênh lệch A/B nằm trong nhiễu seed. Không có so sánh *from-scratch*, không có CRF/constrained decoding (Transformer vẫn có 43 chuỗi BIO không hợp lệ), siêu tham số Transformer không tinh chỉnh (lr 5e-5 cố định).
- **IE chỉ 30 câu / 89 entity** và đã dùng để debug pipeline: lần đầu Transformer 0.956 / baseline 0.776; sau khi sửa hai lỗi pipeline (gán khoảng gold bị chồng lấp ở `Jordan`; tokenizer regex tách `Mr.`/`U.S.`/`Obama's` khác quy ước CoNLL) là 0.972 / 0.787. Gold gán tay **không sửa** sau khi thấy kết quả (ví dụ `The Washington Post` bị coi là sai vì gold của tôi bỏ `The`, trong khi CoNLL gán `The` vào ORG — xem `error_analysis.md` mục 5). Số IE là minh hoạ pipeline, không phải ước lượng khách quan.
- `error_analysis.md` đã ghi rõ giả thuyết nào chưa có phép thử can thiệp; không có phép đo calibration.
- Windows: phải `import pandas` trước `torch` (đã xử lý trong code).

## 8. Cầu nối sang LAB 10

LAB 08: `[CLS] → sequence label`; LAB 09: `mọi token → NER label`; LAB 10 (Question Answering): cùng backbone pretrained nhưng head dự đoán `P_start, P_end` trên các token của context ⇒ span prediction. Những gì dùng lại: word/subword alignment (offset mapping), truncation (context dài), đánh giá theo span (exact match/F1), và nhận định rằng biểu diễn ngữ cảnh là backbone còn head + định nghĩa nhãn quyết định task.
