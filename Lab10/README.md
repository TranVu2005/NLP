# LAB 10 — Extractive Question Answering on SQuAD

Extractive QA trên SQuAD v1.1: `bert-base-cased` + head start/end tự viết (không dùng `AutoModelForQuestionAnswering` hay `pipeline`), căn chỉnh nhãn ký tự → token bằng offset mapping, sliding window, giải mã span, metric EM/F1, baseline keyword-overlap, các thí nghiệm context length / EM-vs-F1 / matching / answer length / nhiễu, error analysis và một ứng dụng Document QA. Mọi số trong tài liệu này lấy từ `results.csv`, `perturbation_results.csv`, `runs/*/result.json` hoặc output của notebook đã chạy.

> **AI-assisted (mục 29).** Toàn bộ code, notebook và các phần viết (`prediction.md`, `calculations.md`, `error_analysis.md`, `learning_check.md`, `reflection.md`) do AI (Claude Code) soạn. Đề cấm giao AI làm thay hand-calculated exercises, prediction, metric interpretation, error analysis, learning check và reflection — **sinh viên cần đọc lại, kiểm chứng và viết lại các phần đó bằng lời của mình trước khi nộp.** `prediction.md` được viết *trước* khi chạy thí nghiệm và không sửa sau khi có kết quả.
> *What was generated:* `qa_model.py`, `baseline_qa.py`, `document_qa.py`, 2 notebook, `test_qa_model.py`, các file markdown. *What was modified:* chưa có sửa đổi nào của sinh viên được ghi nhận. *What was verified:* xem mục 5.

## 1. Cấu trúc & cách chạy

| File | Nội dung |
|---|---|
| `qa_model.py` | Dữ liệu SQuAD v1.1 + split, **EM/F1 tự cài** (chuẩn hoá SQuAD, multiset overlap, max trên gold), `tokenize_qa` (`only_second`, `max_length`, `stride`, overflow, offset mapping), `align_span` (+ các trường hợp biên), `QAFeatures`, `QAModel` (AutoModel + `Linear(d,2)` tự viết), `qa_loss`, giải mã n-best + null score, train/eval, CLI, lưới run |
| `baseline_qa.py` | Baseline: keyword overlap (IDF) → câu ứng viên → span ứng viên (số / cụm viết hoa / cụm từ nội dung); chạy được ở chế độ chỉ trả câu |
| `tokenizer_alignment.ipynb` | **Đã chạy**: Exercise 1–3, `input_ids` / `attention_mask` / `offset_mapping` / `overflow_to_sample_mapping`, alignment (mid-token, truncation, answer ngoài window), sliding window, kiểm chứng mất answer khi cắt không stride (`tokenization_analysis.png`) |
| `qa_finetuning.ipynb` | **Báo cáo chính, đã chạy**: reproducibility, pipeline tensor, baseline, learning curves + Exp 2, Exp 1, Exp 3, Exp 4, Exp 5, nhiễu/đảo thứ tự, error analysis, no-answer, Document QA, ghi `results.csv` |
| `document_qa.py` | `DocumentQA`: (document, question) → answer span, vị trí ký tự, câu chứa đáp án, confidence/margin, top-k, `--null_margin` để từ chối; demo 2 tài liệu (mỗi tài liệu có 1 câu không có đáp án) |
| `results.csv` | 9 dòng: 2 baseline, 3 run context length, 2 seed bổ sung, mean, std (3 seed ở len 384) |
| `perturbation_results.csv` | 8 điều kiện nhiễu/đảo thứ tự: EM/F1 FT + baseline, tỉ lệ prediction rơi vào câu chèn |
| `learning_curves.png`, `em_f1_analysis.png`, `answer_length_analysis.png` | Hình bắt buộc (+ `context_length_analysis.png`, `perturbation_analysis.png`, `no_answer_margin.png`, `tokenization_analysis.png`) |
| `prediction.md`, `calculations.md`, `error_analysis.md`, `learning_check.md`, `reflection.md` | Phần viết |
| `error_analysis_evidence.json`, `error_explanations.json` | 20 mẫu lỗi + giải thích (nguồn của bảng trong `error_analysis.md`) |
| `document_qa_demo.txt` | Output demo Document QA |
| `runs/<tên>/` | `result.json` (config, history, metric val/test, phần cứng), `preds_test.json`; `model.pt` (gitignore) cho run chính; `runs/baseline/` |
| `test_qa_model.py` | 18 test với giá trị tính tay: EM/F1 (`Paris France` → 2/3, `New York` → 0.8), span `Paris`, alignment mid-token / ngoài window / gap, loss = −log 0.6, stride/overflow, mọi answer nằm trọn trong ≥ 1 window, `token_type_ids`, ràng buộc giải mã, shape/padding-invariance, tách câu baseline |

```powershell
# từ thư mục Lab10 (Python 3.14.5, torch 2.12.0+cu126, transformers 5.12.1, datasets, pandas, matplotlib, nbconvert, pytest)
python -m pytest test_qa_model.py -q
python baseline_qa.py                               # baseline (CPU), ghi runs/baseline/
python qa_model.py --run_grid --grad_accum 2        # 5 run (len 384/256/128 + seed 43/44), ~18–23 phút/run trên RTX 3050 6GB
python -m nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 qa_finetuning.ipynb
python document_qa.py --demo
python document_qa.py --context_file doc.txt --question "Who wrote it?"
```

Dữ liệu: lần đầu tải SQuAD v1.1 rồi cache vào `data_cache/` (gitignore). Biến môi trường `LAB10_RUNS_DIR` đổi thư mục run (dùng cho chạy thử), `LAB10_SMALL=1` chạy notebook nhanh.

## 2. Lựa chọn và lý do

- **Dataset & split**: SQuAD v1.1 (87,599 câu hỏi train; 10,570 dev). Train = **20,000 câu hỏi lấy ngẫu nhiên** từ train (398 bài); validation = 1,000 câu từ **các bài (title) không có trong train subset** (44 bài; để val không rò rỉ context); **test = dev chính thức** (48 bài, trung bình 3.29 gold/câu). Chỉ dùng 20k câu vì GPU 6 GB / thời gian: hạn chế thừa nhận ở mục 7.
- **Model**: `bert-base-cased` (107.7M tham số). Head `Linear(768, 2)` = **1,538 tham số** cho start/end logits `[B, T]`, softmax theo T, loss = (CE_start + CE_end)/2; padding mask `−1e4`. Context dài dùng sliding window `max_length` 384, stride 128 (`default_stride = max_length // 3`).
- **Nhãn**: `align_span` đổi `(answer_start, answer_end)` ký tự thành `(start_token, end_token)` chỉ trong vùng context (`sequence_ids`); window không chứa trọn answer có nhãn `[CLS]` (0, 0). Xem `tokenizer_alignment.ipynb` cho mọi trường hợp biên.
- **Giải mã**: top-20 start × top-20 end, `end ≥ start`, tối đa 30 token, chỉ trong context; qua các window lấy span điểm cao nhất; điểm null = `start[CLS] + end[CLS]`. Metric EM/F1 chuẩn SQuAD (lowercase, bỏ dấu câu, bỏ `a/an/the`, multiset overlap, max trên các gold).
- **Huấn luyện**: AdamW (lr 3e-5, weight decay 0.01, không decay bias/LayerNorm), warmup 10% rồi decay tuyến tính, grad-clip 1, fp16 + GradScaler, dropout 0.1, **batch 16 × grad_accum 2 = 32 hiệu dụng**, 2 epoch, padding động. Đánh giá val 4 lần/epoch; **chọn checkpoint theo val F1** (không dùng test); test đánh giá một lần cho checkpoint đó.
- **Baseline** (đề mục 21): IDF học trên các câu train → xếp hạng câu theo overlap từ khoá với câu hỏi → trong câu chọn span ứng viên theo kiểu câu hỏi (số/ngày, cụm viết hoa, cụm nội dung). Chạy một lần, không tinh chỉnh.
- **Seed**: 42 cho mọi cấu hình; 43/44 chỉ cho `len384` để ước lượng nhiễu seed.

## 3. Bảng reproducibility (mục 30)

Chung: Python 3.14.5, torch 2.12.0+cu126, transformers 5.12.1, Windows 11; GPU NVIDIA GeForce RTX 3050 6GB Laptop; `bert-base-cased` pretrained; 107,721,218 tham số (toàn bộ trainable); train 20,000 câu hỏi; val 1,000 câu; test 10,570 câu; `n_best` 20; `max_answer_len` 30; lr 3e-5; batch 16 × grad_accum 2; 2 epoch; warmup 10%; weight decay 0.01; fp16; seed 42/43/44.

| Run | stride | train features | best epoch | thời gian train | VRAM đỉnh | val EM / F1 | **test EM / F1** |
|---|---:|---:|---:|---:|---:|---|---|
| len 384, seed 42 (chính) | 128 | 20,234 | 1.75 | 1,022 s | 3,124 MB | 0.642 / 0.7889 | **0.7522 / 0.8381** |
| len 384, seed 43 | 128 | 20,234 | 2.00 | 1,073 s | 3,133 MB | 0.637 / 0.7753 | 0.7486 / 0.8368 |
| len 384, seed 44 | 128 | 20,234 | 1.50 | 1,069 s | 3,133 MB | 0.635 / 0.7778 | 0.7447 / 0.8329 |
| len 256, seed 42 | 85 | 22,420 | 1.50 | 1,062 s | 2,732 MB | 0.654 / 0.7855 | 0.7497 / 0.8354 |
| len 128, seed 42 | 42 | 43,952 | 2.00 | 1,349 s | 2,310 MB | 0.642 / 0.7786 | 0.7473 / 0.8323 |
| **len 384, mean ± std 3 seed** | | | | 1,054 ± 28 s | | 0.638 / 0.7807 | **0.7485 ± 0.0038 / 0.8359 ± 0.0027** |
| Baseline keyword → câu → span | | | | — | | 0.218 / 0.2977 | 0.2281 / 0.2943 |
| Baseline chỉ trả câu | | | | — | | 0.000 / 0.1669 | 0.0023 / 0.1925 |

Thời gian train là thời gian thuần (không tính đánh giá val). Learning curve từng điểm (train loss, val loss, val EM/F1): `runs/*/result.json`, `learning_curves.png`. Val chỉ 1,000 câu nên val EM/F1 nhiễu hơn test (sai số chuẩn lấy mẫu của EM ≈ ±1.5 điểm).

## 4. Kết quả chính (test = SQuAD v1.1 dev, 10,570 câu)

**Exp 2 — Fine-tune (learning curves)**: train loss 3.36 → 0.85, val loss 1.70 → 1.13 (cực tiểu ở 1.75 epoch), val F1 0.641 → 0.789 sau 2 epoch (8 điểm đánh giá, `learning_curves.png`); val loss tăng nhẹ ở cuối epoch 2 (1.113 → 1.126) trong khi train loss vẫn giảm — dấu hiệu đầu tiên của over-fit nhẹ với 20k câu, nhưng val F1 gần phẳng nên 2 epoch là hợp lý. 3 seed: test EM 0.7485 ± 0.0038, F1 0.8359 ± 0.0027.

**Baseline vs Transformer** (EM / F1): baseline 0.2281 / 0.2943; FT 0.7522 / 0.8381 (seed 42). Baseline chọn đúng *câu chứa đáp án* ở **81.0%** câu test nhưng EM chỉ 0.228 ⇒ khoảng cách nằm ở bước định vị span trong câu, không phải bước tìm câu. Chỉ trả cả câu: EM 0.002 (gần như không bao giờ khớp), F1 0.1925 (precision token 0.12, recall 0.83). Theo kiểu câu hỏi (EM FT / baseline): date 0.866 / 0.505, person 0.830 / 0.345, number 0.791 / 0.289, place 0.740 / 0.284, other 0.711 / 0.139.

**Exp 1 — Context length** (seed 42): xem bảng mục 3 và `error_analysis.md` mục 4. F1 128 / 256 / 384 = 0.8323 / 0.8354 / 0.8381, EM 0.7473 / 0.7497 / 0.7522; thời gian train 1,349 / 1,062 / 1,022 s; VRAM đỉnh 2.3 / 2.7 / 3.1 GB. Chênh từng bước ≈ 1 std giữa seed (EM 0.0038, F1 0.0027) ⇒ **không kết luận được** context dài hơn *thực sự* tốt hơn; chi phí mới là thay đổi rõ (128 chậm hơn 32% vì gấp đôi số feature, nhưng ít VRAM hơn 26%).

**Exp 3 — EM vs F1** (seed 42): EM 0.7522 vs F1 0.8381 (gap 0.086). **1,649 câu (15.6% test; 63% số câu EM = 0)** có EM = 0 nhưng F1 > 0; chỉ 970 câu (9.2%) F1 = 0. Notebook có 12 ví dụ chọn ngẫu nhiên có seed (không cherry-pick), mỗi ví dụ kèm từ thừa/thiếu và P/R/F1: ví dụ `Bellomy Field` vs `Bellomy Field at Santa Clara University` (P 0.33, R 1, F1 0.50), `early 1938` vs `1938` (F1 0.67), `19th` vs `19th century` (F1 0.67), `digital terrestrial` vs `digital terrestrial platform`, `British lace` vs `British lace industry`, `housing` vs `housing assistance`. Token P / R (FT): 0.847 / 0.869. Baseline: 87.6% câu của chế độ chỉ-câu là EM = 0 & F1 > 0 (trả cả câu thì gần như luôn chứa đáp án nhưng không bao giờ khớp).

**Exp 4 — Question/context matching**: F1 theo overlap từ vựng giữa câu hỏi và câu chứa đáp án — thấp / trung bình / cao: 0.785 / 0.859 / 0.894 (chênh 10.9 điểm); model chọn đúng câu 85% / 96% / 98%, baseline 55% / 93% / 97%. Bộ 12 cặp paraphrase viết tay (minh hoạ): FT EM 0.750 (trùng từ) vs 0.833 (paraphrase), baseline F1 0.644 vs 0.472.

**Exp 5 — Độ dài answer** (`answer_length_analysis.png`; FT EM / F1, baseline EM / F1; n):

| Độ dài gold (từ) | n | FT EM | FT F1 | F1 − EM | baseline EM | baseline F1 | độ dài pred TB / gold TB |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 3,600 | 0.798 | 0.849 | 0.051 | 0.345 | 0.366 | 1.77 / 1.00 |
| 2–3 | 4,675 | 0.779 | 0.855 | 0.076 | 0.212 | 0.289 | 2.84 / 2.35 |
| 4–6 | 1,522 | 0.686 | 0.823 | 0.137 | 0.093 | 0.212 | 4.52 / 4.65 |
| 7+ | 773 | 0.511 | 0.716 | 0.205 | 0.045 | 0.153 | 6.12 / 10.41 |

EM giảm đơn điệu theo độ dài (0.80 → 0.51) và khoảng cách F1 − EM tăng (0.05 → 0.21): span dài khó khớp *cả hai* biên, nhưng phần lõi vẫn trùng. Với answer 7+ từ, model cắt ngắn hơn gold (6.1 vs 10.4 từ).

**Nhiễu & đảo thứ tự** (P1, P3, P4; 1,984 câu; `perturbation_results.csv`, bảng đầy đủ + CI ở `error_analysis.md` mục 3): câu ngẫu nhiên ΔF1 +0.0005; câu mồi cùng kiểu −0.0287 [−0.040, −0.019] (5.0% prediction rơi vào câu mồi, baseline 11.2%); nối 2 đoạn trước −0.0294; nối 2 đoạn sau −0.0077; đáp án lên đầu +0.0037; xuống cuối −0.0168; xáo toàn bộ −0.0145.

**Error analysis** (`error_analysis.md`): 2,619 câu EM = 0 — `wrong_location` 970 (37.0%), **wrong_start 768 (29.3%)**, **wrong_end 640 (24.4%)**, wrong_both 221 (8.4%), inner 20. 20 mẫu (5 wrong start, 5 wrong end, 5 wrong span, 5 wrong location) với giải thích. Hơn một nửa lỗi (54.8%) là prediction dài hơn hoặc ngắn hơn gold nhưng chứa nhau.

**No-answer (mở rộng, mục 26)**: model v1.1 luôn trả một span (100% ở cả hai nhóm). Trên 500 câu hỏi ghép context của bài *khác* (không có đáp án), margin `best − null` TB = 5.8 vs 22.0 khi có đáp án; **AUROC = 0.960** cho việc tách hai nhóm; ngưỡng tốt nhất chọn *trên chính tập này* (lạc quan) cho balanced accuracy ≈ 0.894. Không dùng làm ước lượng hiệu năng SQuAD 2.0 thật.

**Document QA** (`document_qa.py`, `document_qa_demo.txt`): 15 câu / 2 tài liệu tự viết. Trong 13 câu có đáp án: **10 đúng**, 1 đúng một phần (`What are the support hours?` → `Monday to Friday`, thiếu `9 a.m. to 6 p.m.`), **2 sai** (`How many days … return an item?` → `5 to 7 business days` thay vì `30 days`; `How long does a refund take?` → `30 days` thay vì `5 to 7 business days` — hai con số cùng kiểu bị đảo, margin vẫn cao 16.2 / 12.8 nên margin không phát hiện được lỗi loại này). Hai câu không có đáp án (`What is the capital of France?`, `Do you ship to Canada?`) vẫn nhận một span nhưng margin thấp (1.49 và 4.67) so với 11.5–28.4 của câu có đáp án — ngưỡng `--null_margin` trong khoảng 5–11 từ chối được cả hai. Đây là demo định tính 15 câu, đánh giá bằng mắt, không phải đánh giá khách quan.

**QA vs RAG** (mục 28): extractive QA = `question + context → span`; RAG = `question → retriever → context → reader/generator → answer`; LAB 10 giả định context đã có; xem `learning_check.md` mục 11 và `reflection.md`.

## 5. Kiểm chứng

- `python -m pytest test_qa_model.py -q` → **18 test đạt** (giá trị tính tay, mục 1).
- `calculations.md` tính tay: Exercise 1–3, loss = −log 0.6, công thức số window `1 + ⌈(N − c)/(c − S)⌉` với `c = L − q − 3`, bảng EM/F1; công thức số window đã kiểm chứng bằng code (0 sai lệch trên 1,000 trường hợp, `tokenizer_alignment.ipynb`).
- **Alignment**: `align_span` và pseudo-code ngây thơ (dùng offset thô không lọc) so với đáp án đúng cho ra **6 nhãn sai** trong các trường hợp biên của notebook (mid-token, answer ở ngoài window, hai gap); cắt context không stride làm mất answer ở **20.5%** câu train ở `max_length = 128`, sliding window đưa về 0 (`tokenizer_alignment.ipynb` mục 5).
- **Chưa** đối chiếu độc lập với script `evaluate-v1.1.py` hay thư viện `evaluate` (cài thêm ngoài `requirements.txt`) — metric chỉ được kiểm bằng test tay; xem mục 7.
- Notebook chạy lại từ cache `runs/` cho cùng số; `LAB10_SMALL=1` chỉ để chạy thử, sẽ ghi đè file kết quả.

## 6. Đối chiếu prediction (`prediction.md` viết trước khi chạy, không sửa)

| | Dự đoán | Kết quả | Đúng? |
|---|---|---|---|
| P1 | Overlap cao → F1 cao hơn (~88 vs ~70); câu mồi cùng kiểu làm F1 giảm 10–25 điểm và phần lớn lỗi mới rơi vào câu mồi; câu ngẫu nhiên giảm < 2; baseline bị mồi lừa nhiều hơn | Overlap cao 0.894 vs thấp 0.785 (chênh 10.9, nhóm thấp *cao hơn* dự đoán); câu mồi **−2.9 điểm** [−4.0, −1.9], chỉ **5.0%** prediction rơi vào câu mồi; câu ngẫu nhiên +0.05; baseline rơi vào mồi 11.2% | Đúng hướng, **sai độ lớn** (−2.9 so với −10…−25; "phần lớn lỗi rơi vào mồi" sai). Có thể do mồi tôi tạo dễ phân biệt hơn mồi đối kháng thật |
| P2 | EM giảm gần đơn điệu theo độ dài (0.75–0.80 → 0.25–0.35); F1 − EM tăng (vài điểm → 20–30) | EM 0.798 / 0.779 / 0.686 / 0.511; F1 − EM 0.051 / 0.076 / 0.137 / 0.205 | Đơn điệu và gap tăng: **đúng**; mức EM ở 2–3, 4–6, 7+ **cao hơn nhiều** dự đoán (0.78 vs 0.50–0.60; 0.69 vs 0.35–0.45; 0.51 vs 0.25–0.35) |
| P3 | Ngẫu nhiên < 2 điểm; distractor −10…−25; context dài giảm khi vượt window | rand2 +0.05; distr1 −2.9; long_pre2 −2.9, long_post2 −0.8 | Câu ngẫu nhiên đúng; distractor **sai độ lớn**; context dài đúng hướng (giảm, nhưng nhỏ) |
| P4 | Câu đáp án lên đầu/xuống cuối ≤ 2 điểm; xáo toàn bộ −3…−6 | lên đầu +0.4; xuống cuối −1.7; xáo toàn bộ −1.45 [−2.4, −0.5] | Phần "lên đầu/xuống cuối ≤ 2" đúng; xáo toàn bộ **nhỏ hơn dự đoán** |
| Exp 1 | F1(128) < F1(256) ≲ F1(384); 384 − 256 < 1; 128 thấp hơn 2–5 điểm; 128 *không* rẻ hơn; 384 chậm nhất | 0.8323 < 0.8354 < 0.8381; 384 − 256 = 0.27; 256 − 128 = 0.31 (không phải 2–5); 128 chậm nhất (1,349 s) | Thứ tự và "384 − 256 < 1" đúng; "128 thấp hơn 2–5 điểm" **sai**; "384 chậm nhất" **sai** (384 nhanh nhất, 1,022 s) — vì 128 gấp đôi số feature. Chênh nằm trong nhiễu seed |
| Exp 2/3 | FT EM 0.68–0.76, F1 0.79–0.85; baseline EM 0.05–0.15, F1 0.15–0.28; ≥ 5 ví dụ EM = 0 & F1 > 0 | FT 0.7522 / 0.8381; baseline 0.2281 / 0.2943; 1,649 câu EM = 0 & F1 > 0 | FT trong khoảng; **baseline cao hơn dự đoán** (bước chọn câu tốt hơn tôi nghĩ) |
| Exp 4 | Overlap cao hơn paraphrase 10–20 điểm; paraphrase tay vẫn đúng đa số nhưng kém | Chênh 10.9 điểm F1 theo overlap; 12 cặp: paraphrase **không kém hơn** (EM 0.833 vs 0.750, n = 12) | Phần overlap đúng; phần paraphrase tay **không xác nhận** (n quá nhỏ) |
| Wrong start/end | wrong-end nhiều hơn wrong-start (55–60%); lỗi vị trí/cả hai biên chiếm phần lớn nhất | wrong_start 768 > wrong_end 640 (wrong-start 54.5% lỗi một biên); wrong_location 970 là nhóm lớn nhất | **Sai** chiều wrong-start/wrong-end; phần "nhóm sai vị trí lớn nhất" đúng |
| Baseline | Chọn đúng câu 55–70%, span kém ⇒ khoảng cách ở định vị span | Chọn đúng câu **81.0%**, EM 0.228 | Ý chính đúng; tỉ lệ chọn đúng câu **cao hơn** dự đoán |

## 7. Hạn chế và lưu ý trung thực

- **Tất cả phần viết/phân tích do AI soạn** (xem đầu file) — cần sinh viên viết lại; phần giải thích nguyên nhân lỗi trong `error_analysis.md` là giả thuyết, chưa có phép thử can thiệp.
- **Train chỉ 20k/87.6k câu hỏi, 2 epoch, không tinh chỉnh siêu tham số**, `grad_accum = 2` (batch hiệu dụng 32 vì GPU 6 GB) ⇒ EM 0.75 / F1 0.84 thấp hơn BERT-base huấn luyện đầy đủ (khoảng 0.81 / 0.88 theo bài báo gốc — không tự đo). Không so sánh trực tiếp với số công bố.
- **Một seed** cho 128/256 và cho toàn bộ phân tích lỗi; chênh giữa các context length (≤ 0.0031 F1) nhỏ hơn hoặc bằng nhiễu seed (std 0.0027 / 0.0038). Kết luận Exp 1 chỉ là "không đo được khác biệt đáng tin về chất lượng; khác biệt về chi phí thì rõ".
- **Val chỉ 1,000 câu** (chọn checkpoint, lệch chút so với test); thời gian train đo trên laptop dùng chung nên chỉ có giá trị tương đối.
- **Metric chưa đối chiếu độc lập** với script SQuAD chính thức (chỉ test tay, mục 5).
- **Bộ paraphrase 12 cặp và Document QA 15 câu là minh hoạ**, tự viết và chỉ chấm bằng mắt; không dùng làm ước lượng khách quan.
- **Perturbation**: câu mồi lấy từ bài khác (không phải đối kháng do người viết) nên −2.9 điểm là ước lượng thấp của độ nhạy; mọi phép đo chỉ trên 1,984 câu.
- **Phân loại lỗi** so token (không so vị trí ký tự) có thể xếp nhầm khi token lặp (mẫu 9053); **nhãn SQuAD nhiễu**: nhiều "lỗi" là gold không đầy đủ hoặc mơ hồ (≥ 7/20 mẫu).
- **No-answer**: model v1.1 không được huấn luyện cho no-answer; AUROC 0.960 là của tập tự dựng (context của bài khác, rất dễ), không phải SQuAD 2.0 thật; ngưỡng chọn trên chính tập đó.
- Windows: phải `import pandas` trước `torch` (đã xử lý trong code); git cần `-c safe.directory=...` (dubious ownership).

## 8. Cầu nối sang LAB 11

LAB 08: `[CLS] → sequence label`; LAB 09: `mọi token → NER label`; LAB 10: `mọi token của context → P_start, P_end` (span prediction); LAB 11 (Retrieval): bỏ giả định "context đã được đưa sẵn" — từ hàng nghìn tài liệu phải *tìm* đoạn chứa đáp án trước (embedding / BM25), rồi mới đọc bằng reader như LAB 10; LAB 12 (RAG): thêm generation để tổng hợp. Điều dùng lại từ lab này: offset mapping, sliding window cho tài liệu dài, điểm null/margin để biết khi nào không có đáp án, và nhận định rằng bước *chọn đúng đoạn* (baseline tìm đúng câu 81%) khác bước *đọc* (EM 0.75).
