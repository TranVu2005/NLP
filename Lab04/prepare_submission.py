"""Xuất hồ sơ tiếng Việt từ kết quả đã chạy; không chỉnh notebook hay model.

Chạy sau run_notebook.py. Chỉ cập nhật vùng AUTO, giữ phần sinh viên viết.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
BEGIN, END = '<!-- AUTO:BEGIN -->', '<!-- AUTO:END -->'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def table(headers, rows):
    def clean(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(clean(v) for v in row)+' |' for row in rows])


def save_region(name, title, body, initial=''):
    path = ROOT/name
    old = path.read_text(encoding='utf-8') if path.exists() else title+'\n\n'+initial
    region = BEGIN+'\n'+body.strip()+'\n'+END
    if BEGIN in old:
        head, tail = old.split(BEGIN, 1)
        _, foot = tail.split(END, 1)
        new = head+region+foot
    else:
        new = old.rstrip()+'\n\n'+region+'\n'
    path.write_text(new, encoding='utf-8')


def main():
    summary_path = ROOT/'run_summary.json'
    s = json.loads(summary_path.read_text(encoding='utf-8'))
    notebook_path = ROOT/'classification.ipynb'
    notebook_before = digest(notebook_path)
    nb = json.loads(notebook_path.read_text(encoding='utf-8'))
    codes = [c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in codes)
    assert [c['execution_count'] for c in codes] == list(range(1,len(codes)+1))
    assert not any(o['output_type']=='error' for c in codes for o in c['outputs'])
    assert len(s['models'])==4 and s['tests_passed']==9
    for field, filename in [('error_examples','error_examples.json'),
                            ('word_order','word_order_demo.json'), ('model_weights','model_weights.json')]:
        assert s[field]==json.loads((ROOT/filename).read_text(encoding='utf-8'))
    with (ROOT/'results.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert all(r['status'] in {'completed','observed'} for r in rows)
    values = {(r['model'],r['split'],r['metric']):r for r in rows}
    for name, run in s['models'].items():
        for split in ['validation','test']:
            for metric,value in run[split].items():
                r=values[name,split,metric]
                assert math.isclose(float(r['value']),value,abs_tol=1e-12)
                assert int(r['vocabulary_size'])==run['vocabulary_size']
                assert math.isclose(float(r['training_time']),run['training_seconds'],abs_tol=1e-12)
    tn,fp = s['baseline_confusion_matrix'][0]
    fn,tp = s['baseline_confusion_matrix'][1]
    baseline=s['models']['tfidf_logistic_unigram']
    assert math.isclose((tn+tp)/(tn+fp+fn+tp),baseline['test']['accuracy'],abs_tol=1e-12)
    assert math.isclose(tp/(tp+fp),baseline['test']['precision'],abs_tol=1e-12)
    assert math.isclose(tp/(tp+fn),baseline['test']['recall'],abs_tol=1e-12)
    assert digest(ROOT/'imdb.zip')==s['archive_sha256']
    origin=f"Nguồn: `run_summary.json`; SHA-256 `{digest(summary_path)}`. Số liệu từ output đã chạy, không fit lại hoặc sửa score."
    parts=[origin, '## 1. Dataset và split',
           f"IMDb: {s['samples']:,} review; {len(s['class_distribution'])} lớp; "
           f"missing text={s['missing_texts']}, missing label={s['missing_labels']}, "
           f"duplicate ngoài lần đầu={s['duplicates']}. Tỷ lệ số mẫu lớp lớn nhất/nhỏ nhất=1.",
           table(['Lớp','Số mẫu','Tỷ lệ (%)'],[[k,v['count'],v['percent']] for k,v in s['class_distribution'].items()]),
           f"Độ dài theo token: {s['length_statistics']}. Không phải độ dài ký tự.",
           table(['Split','Số mẫu','negative','positive'],
                 [[k,v['samples'],v['class_counts']['negative'],v['class_counts']['positive']]
                  for k,v in s['split_summary'].items()]),
           f"Overlap duplicate: `{s['duplicate_overlap']}`. Fit vocabulary/IDF/Word2Vec trên train; "
           f"seed={s['seed']}. Chọn model theo validation F1: `{s['validation_selection']}`.",
           '## 2–3. Baseline và so sánh representation',
           table(['Model','Split','Accuracy','Precision','Recall','F1','Vocabulary','Thời gian (s)'],
                 [[name,split,*[f"{run[split][m]:.9f}" for m in ['accuracy','precision','recall','f1']],
                   run['vocabulary_size'],f"{run['training_seconds']:.6f}"]
                  for name,run in s['models'].items() for split in ['validation','test']]),
           'Lớp positive dùng cho binary Precision/Recall/F1. Thời gian Word2Vec gồm train embedding, tạo vector và fit LR; '
           'thời gian TF-IDF gồm fit pipeline. Không coi các thời gian là riêng thời gian classifier.',
           '**Chênh lệch test so với TF-IDF unigram/LR (điểm phần trăm):**',
           table(['Model','ΔAccuracy','ΔPrecision','ΔRecall','ΔF1'],
                 [[name,*[f"{100*(run['test'][m]-baseline['test'][m]):+.6f}"
                           for m in ['accuracy','precision','recall','f1']]]
                  for name,run in s['models'].items() if name!='tfidf_logistic_unigram']),
           'Các chênh lệch mô tả những cấu hình đã chạy; không xác lập nguyên nhân về ngữ nghĩa hoặc khả năng tổng quát cho dataset khác.',
           '## 4. Hai câu demo phủ định',
           table(['Câu','Representation','OOV','Prediction','P(positive)'],
                 [[r['text'],r['representation'],r['oov_tokens'],r['prediction'],
                   f"{r['probabilities']['positive']:.15f}"] for r in s['word_order']
                  if 'probabilities' in r]),
           'TF-IDF features và toàn bộ averaged vectors 100 chiều được lưu trong `word_order_demo.json` và notebook. '
           'Hai câu thay đổi cả số token và có/không có “not”, không phải hoán vị cùng tập từ; '
           'không đủ để chứng minh representation hiểu phủ định hay đo riêng word order.',
           '## 5. Unigram so với unigram + bigram']
    bigram=s['models']['tfidf_logistic_unigram_bigram']
    parts += [table(['Đại lượng','Unigram','Uni + bigram','Chênh lệch'],
                    [['Vocabulary',baseline['vocabulary_size'],bigram['vocabulary_size'],
                      bigram['vocabulary_size']-baseline['vocabulary_size']],
                     ['F1 test',f"{baseline['test']['f1']:.12f}",f"{bigram['test']['f1']:.12f}",
                      f"{bigram['test']['f1']-baseline['test']['f1']:+.12f}"],
                     ['Fit time (s)',f"{baseline['training_seconds']:.6f}",f"{bigram['training_seconds']:.6f}",
                      f"{bigram['training_seconds']-baseline['training_seconds']:+.6f}"]]),
              f"Feature bigram thật và coefficients: `{s['model_weights']['bigram_features']}`. "
              'Cap vocabulary tăng từ 50.000 lên 100.000, nên không quy toàn bộ chênh lệch F1 cho riêng n-gram.',
              '## Evaluation — baseline test',
              f"Confusion matrix `{s['baseline_confusion_matrix']}`; hàng true/cột predicted theo negative, positive. "
              f"TN={tn}, FP={fp}, FN={fn}, TP={tp}; tổng lỗi={fp+fn}.",
              table(['Class','Precision','Recall','F1','Support'],
                    [[label,*[f"{s['baseline_classification_report'][label][m]:.9f}"
                              for m in ['precision','recall','f1-score']],
                      int(s['baseline_classification_report'][label]['support'])]
                     for label in ['negative','positive']]),
              '![Confusion matrix](confusion_matrix.png)',
              '## Giới hạn và trạng thái',
              'Archive không ghi upstream/release và CSV không có official split. Chia lại có kiểm soát duplicate. '
              'Chưa chạy ablation stopword hoặc giữ nguyên n-gram rồi thay cap vocabulary. '
              'Không dùng test để điều chỉnh cấu hình; không báo độ tin cậy thống kê từ một seed. '
              'Các câu trả lời tham khảo AI phải được phân biệt với phần cá nhân theo W4 trang 12.']
    save_region('experiment_report.md','# Báo cáo bằng chứng LAB 04','\n\n'.join(parts))
    errors=[origin, 'Model: TF-IDF unigram + Logistic Regression; test. '
            'Lấy 3 mẫu đúng và 3 mẫu sai đầu tiên theo thứ tự test, không đại diện toàn bộ loại lỗi.',
            table(['Mẫu','Kết quả','True','Predicted','P(negative)','P(positive)'],
                  [[i,e['kind'],e['true_label'],e['predicted_label'],
                    f"{e['predicted_scores']['negative']:.15f}",f"{e['predicted_scores']['positive']:.15f}"]
                   for i,e in enumerate(s['error_examples'],1)])]
    for i,e in enumerate(s['error_examples'],1):
        errors += [f"## Mẫu {i} — {'đúng' if e['kind']=='correct' else 'sai'}",
                   f"True=`{e['true_label']}`; predicted=`{e['predicted_label']}`; "
                   f"score lớp dự đoán={e['confidence']:.15f}.",
                   '**Văn bản thực tế đầy đủ:**\n\n'+'\n'.join('> '+line for line in e['text'].splitlines()),
                   '**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**\n\n'+
                   table(['Feature','Contribution'],[[term,f'{value:.15f}'] for term,value in e['supporting_features']]),
                   'Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; '
                   'chỉ 5 feature nên tổng không được coi là toàn logit.']
    initial=('W4 trang 12 yêu cầu sinh viên tự chọn nguyên nhân lỗi. Bảng tự động dưới đây chỉ xuất bằng chứng, chưa phải phân tích hoàn chỉnh.\n\n'
             '## Phân tích do sinh viên viết\n\n'
             '- Mẫu sai 4: trích đoạn quyết định, nguyên nhân và bằng chứng: …\n'
             '- Mẫu sai 5: trích đoạn quyết định, nguyên nhân và bằng chứng: …\n'
             '- Mẫu sai 6: trích đoạn quyết định, nguyên nhân và bằng chứng: …\n'
             '- So sánh với 3 mẫu đúng: …\n'
             '- Failure quan trọng nhất và phép thử tiếp theo: …\n\n'
             'Các loại lỗi gợi ý trong W4: lexical ambiguity, negation, sarcasm, short text, rare word, domain term, data noise, class imbalance. '
             'Không gán loại lỗi nếu chưa có bằng chứng trong văn bản/model.\n')
    save_region('error_analysis.md','# Bằng chứng và phân tích lỗi LAB 04','\n\n'.join(errors),initial)
    w=s['model_weights']
    weights=[origin, f"Model baseline; label mapping `{w['label_mapping']}`, classes `{w['classes']}`. "
             'Coefficient dương hướng về positive, âm hướng về negative; là hệ số logit, không phải xác suất.',
             '## Top positive features',table(['Feature','Coefficient'],w['highest_positive']),
             '## Top negative features',table(['Feature','Coefficient'],w['most_negative']),
             '## Feature bigram',table(['Feature','Coefficient'],w['bigram_features'].items()),
             'Bigram thuộc model unigram+bigram riêng; không so coefficient tuyệt đối giữa hai model như cùng một thang ảnh hưởng.']
    save_region('model_interpretability.md','# Kiểm tra trọng số mô hình','\n\n'.join(weights),
                '## Sinh viên tự diễn giải\n\nFeature nào phù hợp sentiment và feature nào cần kiểm tra ngữ cảnh? Dẫn review thực tế: …\n')
    assert digest(notebook_path)==notebook_before, 'Notebook bị thay đổi'
    names=['W4.pdf','23001958_Tran_Xuan_Vu_Lab04.pdf','imdb.zip','classification.ipynb',
           'preprocessing.py','test_lab04.py','run_summary.json','results.csv','confusion_matrix.png',
           'class_distribution.png','error_examples.json','word_order_demo.json','model_weights.json',
           'calculations.md','prediction.md','error_analysis.md','reflection.md','experiment_report.md',
           'model_interpretability.md','learning_check.md','AI_CONTRIBUTION.md','rubric_checklist.md',
           'README.md','requirements.txt','run_notebook.py','annotate_results.py','prepare_submission.py',
           'notebook_answers.json']
    manifest={'code_cells':len(codes),'notebook_error_outputs':0,'completed_models':len(s['models']),
              'results_rows':len(rows),'tests_recorded_passed':s['tests_passed'],
              'personal_sections_complete':False,'sha256':{n:digest(ROOT/n) for n in names}}
    (ROOT/'submission_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sha256'},ensure_ascii=False))


if __name__=='__main__':
    main()
