"""Tổng hợp Markdown tiếng Việt từ kết quả thật, không sửa code/output."""
from pathlib import Path
import json
import hashlib
import uuid

ROOT = Path(__file__).resolve().parent
ANSWERS = {}


def metrics_table(run):
    result = "| Tập | Accuracy | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n"
    for split in ["validation", "test"]:
        m = run[split]
        result += f"| {split} | {m['accuracy']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['f1']:.9f} |\n"
    return result


def questions(evidence, question):
    text = f"\n\n**Bằng chứng:** {evidence}\n\n**Câu hỏi cho sinh viên:** {question}\n"
    if question in ANSWERS:
        text += "\n**Trả lời tham khảo (AI hỗ trợ, dựa trên kết quả thực nghiệm):**\n\n" + ANSWERS[question] + "\n"
    return text


def main():
    global ANSWERS
    summary = json.loads((ROOT / "run_summary.json").read_text(encoding="utf-8"))
    answer_file = ROOT / "notebook_answers.json"
    if answer_file.exists():
        payload = json.loads(answer_file.read_text(encoding="utf-8"))
        if payload["source_summary_sha256"] == hashlib.sha256((ROOT / "run_summary.json").read_bytes()).hexdigest():
            ANSWERS = payload["answers"]
        else:
            print("Kết quả thực nghiệm đã đổi; cần đối chiếu lại câu trả lời trước khi chèn.")
    notebook_path = ROOT / "classification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    before = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    before = json.dumps(before, ensure_ascii=False, sort_keys=True)
    models = summary["models"]
    baseline = models["tfidf_logistic_unigram"]
    w2v = models["average_word2vec_logistic"]
    svm = models["tfidf_linear_svm_unigram"]
    bigram = models["tfidf_logistic_unigram_bigram"]
    assert len(models) == 4 and summary["tests_passed"] == 9
    cm = summary["baseline_confusion_matrix"]
    replacements = {}

    text = "## Thí nghiệm 1 - Khảo sát tập dữ liệu\n\n**Số liệu quan sát được:**\n\n"
    text += f"- {summary['samples']:,} mẫu, 2 lớp: negative và positive đều 25.000 (50%). Tỷ lệ lớp lớn nhất/nhỏ nhất = 1; chênh lệch = 0 điểm phần trăm.\n"
    text += f"- Văn bản thiếu: {summary['missing_texts']}; nhãn thiếu: {summary['missing_labels']}; dòng trùng ngoài lần đầu: {summary['duplicates']}.\n"
    stats = summary["length_statistics"]
    text += f"- Độ dài theo token khoảng trắng: trung bình {stats['average']}, trung vị {stats['median']}, min={stats['min']}, max={stats['max']}.\n"
    for split, info in summary["split_summary"].items():
        text += f"- {split}: {info['samples']:,} mẫu; phân bố nhãn {info['class_counts']}.\n"
    text += f"- Seed={summary['seed']}; trùng văn bản giữa các tập: {summary['duplicate_overlap']}.\n"
    text += "\nLần chạy đầy đủ dùng StratifiedGroupKFold 14/3/3 trên 20 fold, giữ các review trùng trong cùng tập. Kích thước khác lần chỉ kiểm tra bằng phương pháp dự phòng trước đây. Không downsample.\n"
    text += questions("Output khảo sát và split vừa chạy trên toàn bộ IMDb được cung cấp.", "Counts và tỷ lệ thực tế cho thấy mức mất cân bằng nào?")
    replacements["## Thí nghiệm 1"] = text

    text = "## Thí nghiệm 2 - TF-IDF + Logistic Regression\n\n**Số liệu quan sát được:**\n\n"
    text += f"Vocabulary={baseline['vocabulary_size']:,}; thời gian fit={baseline['training_seconds']:.6f} giây. Lớp dương là positive.\n\n"
    text += metrics_table(baseline)
    text += f"\nConfusion matrix test (hàng thực tế, cột dự đoán; negative, positive): `{cm}`.\n"
    text += questions("Pipeline fit vocabulary trên train, validation/test chỉ transform. Không chọn siêu tham số từ test.", "Chỉ số nào phù hợp nhất với loại lỗi bạn cần ưu tiên?")
    replacements["## Thí nghiệm 2"] = text

    text = "## Thí nghiệm 3 - Average Word2Vec + Logistic Regression\n\n**Số liệu quan sát được:**\n\n"
    text += f"Vocabulary={w2v['vocabulary_size']:,}; thời gian train Word2Vec, tạo vector và fit LR={w2v['training_seconds']:.6f} giây.\n\n"
    text += metrics_table(w2v)
    text += "\nSkip-gram; vector_size=100, window=5, min_count=2, epochs=10, workers=1, seed=42. Chỉ train Word2Vec trên train; tài liệu không có token đã biết nhận vector 0.\n"
    text += questions("Kết quả mới đo trên cùng split với TF-IDF.", "Những chênh lệch đo được hỗ trợ kết luận nào về hai biểu diễn?")
    replacements["## Thí nghiệm 3 - Trạng thái"] = text
    replacements["## Thí nghiệm 3 — Trạng thái"] = text
    replacements["## Thí nghiệm 3 - Average Word2Vec"] = text

    text = "## Thí nghiệm 3 - So sánh biểu diễn và classifier\n\n**Số liệu quan sát được:**\n\n"
    text += "| Mô hình | Tập | Accuracy | Precision | Recall | F1 |\n|---|---|---:|---:|---:|---:|\n"
    for name, run in [("TF-IDF/LR", baseline), ("Average Word2Vec/LR", w2v), ("TF-IDF/Linear SVM", svm)]:
        for split in ["validation", "test"]:
            m = run[split]
            text += f"| {name} | {split} | {m['accuracy']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['f1']:.9f} |\n"
    for name, run in [("Word2Vec/LR", w2v), ("Linear SVM", svm)]:
        for split in ["validation", "test"]:
            deltas = {m: run[split][m] - baseline[split][m] for m in ["accuracy", "precision", "recall", "f1"]}
            text += f"\nChênh lệch {name} - TF-IDF/LR ({split}): " + "; ".join(f"{k}={v:+.9f}" for k, v in deltas.items()) + ".\n"
    text += f"\nCấu hình có validation F1 cao nhất trong bốn cấu hình định trước: `{summary['validation_selection']}`. Không suy ra nguyên nhân ngữ nghĩa từ các điểm số này.\n"
    text += questions("Cùng train/validation/test; cấu hình đã xác định trước khi đọc test metrics.", "Chênh lệch validation có đủ để quyết định cho nhiệm vụ của bạn không?")
    replacements["## Thí nghiệm 3 - So sánh"] = text
    replacements["## Thí nghiệm 3 — So sánh"] = text

    text = "### Kết quả Linear SVM\n\n**Số liệu quan sát được:**\n\n"
    text += f"Vocabulary={svm['vocabulary_size']:,}; thời gian fit={svm['training_seconds']:.6f} giây.\n\n" + metrics_table(svm)
    text += questions("Linear SVM đã chạy, không bị bỏ qua. Vectorizer chỉ fit trên train.", "Bạn đánh giá sự khác biệt với LR thế nào theo metric đo được?")
    replacements["### Trạng thái chạy"] = text
    replacements["### Kết quả Linear SVM"] = text

    demos = summary["word_order"]
    text = "## Thí nghiệm 4 - Thứ tự từ và phủ định\n\n**Số liệu quan sát được:**\n\n"
    text += "| Câu | Biểu diễn | Token | OOV | Dự đoán | P(positive) |\n|---|---|---:|---:|---|---:|\n"
    for row in demos:
        text += f"| `{row['text']}` | {row['representation']} | {len(row['tokens'])} | {len(row['oov_tokens'])} | {row['prediction']} | {row['probabilities']['positive']:.9f} |\n"
    for row in demos:
        if row["representation"] == "TF-IDF unigram":
            text += f"\nTF-IDF `{row['text']}`: `{row['tfidf_features']}`.\n"
        else:
            text += f"\nVector trung bình `{row['text']}`: 8 tọa độ đầu `{row['averaged_vector'][:8]}`; chuẩn L2={row['vector_l2_norm']:.9f}; toàn bộ 100 tọa độ có trong output và word_order_demo.json.\n"
    cosine = next(r["cosine_S1_S2"] for r in demos if r["representation"] == "Average Word2Vec")
    text += f"\nCosine giữa hai vector trung bình: **{cosine:.12f}**. OOV từng câu được liệt kê trong JSON. Hai câu chỉ là minh họa, không phải evaluation định lượng; average không trực tiếp mô hình hóa thứ tự hoặc phạm vi phủ định.\n"
    text += questions("TF-IDF features, vector và xác suất được tính từ các model vừa train.", "Kết quả này cho biết gì về token not, và điều gì về composition vẫn chưa được kiểm tra?")
    replacements["## Thí nghiệm 4"] = text

    text = "## Thí nghiệm 5 - Unigram và unigram + bigram\n\n**Số liệu quan sát được:**\n\n"
    text += "| Cấu hình | Vocabulary | Thời gian fit (s) | F1 validation | F1 test |\n|---|---:|---:|---:|---:|\n"
    for label, run in [("Unigram", baseline), ("Unigram + bigram", bigram)]:
        text += f"| {label} | {run['vocabulary_size']:,} | {run['training_seconds']:.6f} | {run['validation']['f1']:.9f} | {run['test']['f1']:.9f} |\n"
    text += f"\nChênh lệch bigram - unigram: vocabulary={bigram['vocabulary_size']-baseline['vocabulary_size']:+d}; thời gian={bigram['training_seconds']-baseline['training_seconds']:+.6f}s; F1 validation={bigram['validation']['f1']-baseline['validation']['f1']:+.9f}; F1 test={bigram['test']['f1']-baseline['test']['f1']:+.9f}.\n"
    text += f"\nFeature có thật và coefficient: `{summary['model_weights']['bigram_features']}`.\n"
    text += "\nCùng split và Logistic Regression. Cấu hình đã định trước dùng max_features=50.000 ở unigram và 100.000 ở bigram; khác biệt cap cũng ảnh hưởng phép so sánh, nên không quy toàn bộ chênh lệch cho n-gram.\n"
    text += questions("Các feature và hệ số lấy từ vocabulary thật, không ghi cứng.", "Số liệu có hỗ trợ sử dụng cấu hình bigram này không?")
    replacements["## Thí nghiệm 5"] = text

    report = summary["baseline_classification_report"]
    text = "## Đánh giá - Ma trận nhầm lẫn và chỉ số theo lớp\n\n**Số liệu quan sát được:**\n\n"
    text += f"Ma trận test: `{cm}`. Hàng là nhãn thực tế, cột là nhãn dự đoán; thứ tự negative, positive. TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}.\n\n"
    text += "| Lớp | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n"
    for label in ["negative", "positive"]:
        m = report[label]
        text += f"| {label} | {m['precision']:.9f} | {m['recall']:.9f} | {m['f1-score']:.9f} | {int(m['support'])} |\n"
    text += "\n![Ma trận nhầm lẫn](confusion_matrix.png)\n"
    text += questions("Ma trận và classification_report lấy từ dự đoán test mới của baseline.", "FP và FN phản ánh loại lỗi nào, và loại nào quan trọng hơn đối với ứng dụng?")
    replacements["## Đánh giá"] = text

    text = "## Bằng chứng cho phân tích lỗi\n\n**Số liệu quan sát được:** 3 review đúng và 3 review sai lấy từ test. Full text, xác suất từng lớp và feature hỗ trợ có trong error_examples.json.\n"
    for index, row in enumerate(summary["error_examples"], 1):
        preview = row["text_preview"].replace("\n", " ")
        text += f"\n### Ví dụ {index}: {'đúng' if row['kind']=='correct' else 'sai'}\n\n"
        text += f"Nhãn thật={row['true_label']}; dự đoán={row['predicted_label']}; confidence={row['confidence']:.9f}; xác suất={row['predicted_scores']}.\n\n"
        text += "> " + preview + "\n\n"
        text += f"Đóng góp feature hỗ trợ: `{row['supporting_features']}`.\n"
    text += questions("Văn bản và score đều từ lần dự đoán test mới; diễn giải tham khảo chỉ dựa trên bằng chứng trong review.", "Bằng chứng trong review hỗ trợ cách giải thích của bạn thế nào?")
    replacements["## Bằng chứng cho phân tích lỗi"] = text

    weights = summary["model_weights"]
    text = "## Khả năng diễn giải mô hình - Hệ số feature\n\n**Số liệu quan sát được:** ánh xạ negative=0, positive=1; classes của LR là `[negative, positive]`. Coefficient dương hướng về positive, âm hướng về negative.\n"
    for key, title in [("highest_positive", "20 hệ số dương lớn nhất"), ("most_negative", "20 hệ số âm nhỏ nhất")]:
        text += f"\n### {title}\n\n| Feature | Coefficient |\n|---|---:|\n"
        for token, coefficient in weights[key]:
            text += f"| `{token}` | {coefficient:.12f} |\n"
    text += f"\nCác term cần kiểm tra (None nghĩa là không thuộc vocabulary unigram): `{weights['requested_features']}`. Hệ số bigram: `{weights['bigram_features']}`. Giá trị đầy đủ trong model_weights.json.\n"
    text += questions("Trích trực tiếp feature names và coef_ của model đã fit.", "Các coefficient đo được hỗ trợ diễn giải nào, và giới hạn của cách diễn giải đó là gì?")
    replacements["## Khả năng diễn giải mô hình"] = text

    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        for prefix, replacement in replacements.items():
            if source.startswith(prefix):
                cell["source"] = replacement.splitlines(keepends=True)
                break
    # Hai câu hỏi được render trong output của code cũng được trả lời riêng.
    notebook["cells"] = [c for c in notebook["cells"] if not c.get("metadata", {}).get("codex_output_question_answer")]
    updated = []
    for cell in notebook["cells"]:
        updated.append(cell)
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        answer = None
        if source.startswith("if baseline is not None:\n    from sklearn.metrics import classification_report"):
            answer = "**Câu hỏi từ output:** Chỉ số nào thể hiện chi phí bỏ sót review dương tính?\n\n"
            answer += f"**Trả lời tham khảo (AI hỗ trợ):** Recall của positive thể hiện tỷ lệ review positive được nhận diện đúng; FNR=1-Recall thể hiện tỷ lệ bỏ sót. Baseline có FN={cm[1][0]} trên {cm[1][0]+cm[1][1]} review positive, FNR={1-baseline['test']['recall']:.9f}, Recall={baseline['test']['recall']:.9f}. Đây là số đếm/tỷ lệ lỗi; chi phí thực tế còn cần một giá trị chi phí cho mỗi FN do ứng dụng quy định.\n"
        elif source.startswith("comparison=["):
            answer = "**Câu hỏi từ output:** Biểu diễn nào có kết quả validation tốt hơn theo các số đo này?\n\n"
            answer += f"**Trả lời tham khảo (AI hỗ trợ):** Trong hai model ở output này, TF-IDF/LR có validation Accuracy={baseline['validation']['accuracy']:.9f}, F1={baseline['validation']['f1']:.9f}; Average Word2Vec/LR có Accuracy={w2v['validation']['accuracy']:.9f}, F1={w2v['validation']['f1']:.9f}. TF-IDF/LR cao hơn ở cả Accuracy, Precision, Recall và F1; chênh lệch F1={(baseline['validation']['f1']-w2v['validation']['f1'])*100:.6f} điểm phần trăm. Kết luận chỉ áp dụng cho cấu hình và split đã chạy; chưa xác định nguyên nhân ngữ nghĩa.\n"
        if answer:
            updated.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                            "metadata": {"codex_output_question_answer": True},
                            "source": answer.splitlines(keepends=True)})
    notebook["cells"] = updated
    after = json.dumps([c for c in notebook["cells"] if c["cell_type"] == "code"], ensure_ascii=False, sort_keys=True)
    assert before == after
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = "# LAB 04 - Phân loại văn bản\n\n"
    readme += "Đã chạy đủ cả 4 cấu hình bằng kernel Jupyter mới. 9/9 unit tests đạt; không có thí nghiệm bị bỏ qua. Markdown và các tệp chi tiết được tổng hợp từ output thật.\n\n"
    readme += f"Corpus: {summary['samples']:,} review IMDb từ imdb.zip có sẵn trong workspace; CSV không chứa official split. Archive không ghi upstream hoặc release. SHA-256: `{summary['archive_sha256']}`.\n\n"
    readme += f"Seed={summary['seed']}; nhãn negative=0, positive=1; binary metrics dùng positive. Chia theo nhóm văn bản trùng bằng StratifiedGroupKFold 14/3/3 của 20 fold: `{summary['split_summary']}`. Trùng chéo: `{summary['duplicate_overlap']}`. Không downsample.\n\n"
    readme += "## Kết quả test\n\n| Mô hình | Accuracy | Precision | Recall | F1 | Vocabulary | Thời gian (s) |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, run in models.items():
        m = run["test"]
        readme += f"| {name} | {m['accuracy']:.9f} | {m['precision']:.9f} | {m['recall']:.9f} | {m['f1']:.9f} | {run['vocabulary_size']} | {run['training_seconds']:.6f} |\n"
    readme += f"\nCấu hình chọn theo validation F1: `{summary['validation_selection']}`. Test chỉ đánh giá những cấu hình đã xác định trước; không chỉnh model theo test.\n\n"
    readme += f"Confusion matrix baseline test: `{cm}`; hàng thực tế/cột dự đoán theo negative, positive. FP={cm[0][1]}, FN={cm[1][0]}.\n\n"
    readme += "## Cấu hình và giới hạn\n\nTF-IDF dùng min_df=2; unigram có max_features=50.000, unigram+bigram=100.000. So sánh dùng cùng split và Logistic Regression, nhưng khác cap vocabulary nên không quy toàn bộ chênh lệch cho n-gram. Word2Vec Skip-gram: vector_size=100, window=5, min_count=2, epochs=10, workers=1, seed=42; PYTHONHASHSEED=42. Chỉ học từ train; document không có từ đã biết nhận vector 0. Thời gian Word2Vec gồm train embedding, tạo vector và fit LR. Hai câu demo không thay thế đánh giá định lượng trên test.\n\n"
    code_count = sum(c["cell_type"] == "code" for c in notebook["cells"])
    readme += f"## Tệp\n\n- classification.ipynb: các phần thí nghiệm, {code_count} code cells, output và Markdown tiếng Việt.\n"
    readme += "- preprocessing.py và test_lab04.py: helper và 9 numerical/preprocessing tests.\n- results.csv: metrics validation/test, confusion matrices, demo và dataset facts; input_text xác định câu demo.\n- class_distribution.png và confusion_matrix.png: biểu đồ thực tế, ảnh trạng thái cũ đã được thay.\n- run_summary.json: đầy đủ kết quả, split, phiên bản, matrix, report từng lớp và model selection theo validation.\n- word_order_demo.json: TF-IDF, hai vector Word2Vec 100 chiều, OOV và xác suất.\n- error_examples.json: 3 đúng/3 sai, full text, nhãn, xác suất và feature hỗ trợ.\n- notebook_answers.json: câu trả lời tham khảo do AI hỗ trợ, gắn với SHA-256 của kết quả thực nghiệm để tránh dùng câu trả lời cũ khi kết quả đổi.\n- model_weights.json: top-20 hệ số dương/âm, ánh xạ nhãn và term/bigram thật.\n- requirements.txt: phiên bản thư viện đã chạy.\n- run_notebook.py và annotate_results.py: chạy kernel mới và cập nhật Markdown theo output.\n\n"
    readme += "## Chạy lại\n\nTrong môi trường Python phù hợp tại Lab04:\n\n```powershell\npython -m pip install -r requirements.txt\npython run_notebook.py\n```\n\nTrên workspace hiện tại, dùng Python đóng gói tại `C:\\Users\\Admin\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe`; runner tự nạp thư viện trong `.lab01-deps` dùng chung. Notebook báo lỗi nếu thiếu thư viện bắt buộc và assert đủ 4 model trước khi export.\n\n"
    readme += f"## AI hỗ trợ và kiểm chứng\n\nTool: OpenAI Codex. Mục đích: code/debug/thực nghiệm/đồ thị và tổng hợp Markdown từ output. Đã xác minh: {summary['tests_passed']}/9 tests, kernel mới chạy toàn bộ notebook, 4 model, CSV và dữ liệu chi tiết. Không viết thay reflection hoặc nguyên nhân lỗi cá nhân. Phiên bản: `{summary['versions']}`.\n"
    readme += "\n## Hồ sơ theo W4\n\n"
    readme += "- calculations.md và prediction.md: chép từ PDF bài viết tay đã có; không tạo dự đoán mới sau khi biết kết quả. Confidence và thời điểm trước experiment chưa được xác minh.\n"
    readme += "- experiment_report.md: báo cáo 5 experiments, metrics, chênh lệch và giới hạn từ output thật.\n"
    readme += "- error_analysis.md: văn bản đầy đủ, nhãn, score và contribution cho 3 đúng/3 sai; sinh viên tự viết nguyên nhân trong phần riêng.\n"
    readme += "- model_interpretability.md: top-20 hệ số dương/âm, label mapping và coefficient bigram.\n"
    readme += "- reflection.md và learning_check.md: khung cho phần sinh viên tự hoàn thiện theo W4 trang 12–13.\n"
    readme += "- AI_CONTRIBUTION.md và rubric_checklist.md: khai báo hỗ trợ AI và đối chiếu 100 điểm. Các câu trả lời tham khảo AI trong notebook không được coi là bài cá nhân.\n"
    readme += "- prepare_submission.py và submission_manifest.json: kiểm tra JSON/CSV, xuất báo cáo và SHA-256; giữ nguyên notebook/code/output.\n\n"
    readme += "Sau khi chạy notebook, chạy `python prepare_submission.py`. Runner cũng tự gọi script này khi có file. Các vùng AUTO được cập nhật; phần sinh viên viết bên ngoài vùng AUTO được giữ nguyên. Bài chưa hoàn chỉnh phần cá nhân khi còn dấu … trong error analysis/reflection/learning check.\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("Updated Vietnamese evidence Markdown from completed models; code/output unchanged.")


if __name__ == "__main__":
    main()
