"""LAB 09 - Baseline NER (muc 13): token features (cua so +-2 tu) -> TF-IDF -> Logistic Regression.

    word + features tu/ngu canh  ->  DictVectorizer  ->  TfidfTransformer  ->  LogisticRegression (9 nhan BIO)

Moi token duoc phan loai DOC LAP (khong CRF, khong rang buoc BIO, khong dung POS/chunk tags co san trong CoNLL)
nen day la baseline "classifier don gian dua tren word features": chi nhin tu hien tai + vai tu lan can duoi dang
feature roi rac, khong co bieu dien ngu canh. Dung cung train/val/test CoNLL-2003 va cung metric entity-level
(`ner_core.entity_metrics`) nhu Transformer.

CLI:  python baseline_ner.py            # chon C theo val entity-F1, danh gia test, luu runs/baseline/
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

import ner_core as nc

BASELINE_DIR = nc.RUNS_DIR / "baseline"
C_GRID = (3.0, 10.0, 30.0)


def word_shape(w: str) -> str:
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)       # nen lap: "Xxxxxx" -> "Xxx"


def token_features(words: Sequence[str], i: int) -> dict[str, float]:
    w = words[i]
    f = {"bias": 1.0, "w.lower=" + w.lower(): 1.0, "w.shape=" + word_shape(w): 1.0,
         "w.isupper": float(w.isupper()), "w.istitle": float(w.istitle()), "w.isdigit": float(w.isdigit()),
         "w.hyphen": float("-" in w), "w.len>10": float(len(w) > 10), "pos0": float(i == 0), "posN": float(i == len(words) - 1)}
    for k in (1, 2, 3):
        f[f"w.suf{k}={w[-k:].lower()}"] = 1.0
        f[f"w.pre{k}={w[:k].lower()}"] = 1.0
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(words):
            v = words[j]
            f[f"{off}:lower={v.lower()}"] = 1.0
            f[f"{off}:shape={word_shape(v)}"] = 1.0
            f[f"{off}:istitle"] = float(v.istitle())
            f[f"{off}:isupper"] = float(v.isupper())
        else:
            f[f"{off}:pad"] = 1.0
    if i > 0:
        f[f"bigram={words[i - 1].lower()}|{w.lower()}"] = 1.0
    return {k: v for k, v in f.items() if v != 0.0}      # bo feature = 0 (log(0) trong sublinear TF-IDF -> -inf)


def sentence_features(sentences: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    return [token_features(s, i) for s in sentences for i in range(len(s))]


class BaselineNER:
    def __init__(self, C: float = 10.0, max_iter: int = 300):
        self.vec = DictVectorizer(sparse=True)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self.C = C

    def fit(self, sentences: Sequence[Sequence[str]], tags: Sequence[Sequence[str]]) -> "BaselineNER":
        X = self.tfidf.fit_transform(self.vec.fit_transform(sentence_features(sentences)))
        y = [nc.LABEL2ID[t] for s in tags for t in s]
        self.clf.fit(X, y)
        return self

    def predict_proba(self, sentences: Sequence[Sequence[str]]) -> np.ndarray:
        X = self.tfidf.transform(self.vec.transform(sentence_features(sentences)))
        p = np.zeros((X.shape[0], len(nc.LABELS)), dtype=np.float32)
        p[:, self.clf.classes_] = self.clf.predict_proba(X)
        return p

    def predict(self, sentences: Sequence[Sequence[str]]) -> tuple[list[list[str]], list[list[float]]]:
        p = self.predict_proba(sentences)
        out_t, out_c, k = [], [], 0
        for s in sentences:
            pp = p[k:k + len(s)]
            out_t.append([nc.LABELS[int(j)] for j in pp.argmax(1)])
            out_c.append([float(x) for x in pp.max(1)])
            k += len(s)
        return out_t, out_c

    @property
    def n_features(self) -> int:
        return len(self.vec.vocabulary_)


def run_baseline(force: bool = False, verbose: bool = True) -> dict[str, Any]:
    if (BASELINE_DIR / "result.json").exists() and not force:
        return json.loads((BASELINE_DIR / "result.json").read_text(encoding="utf8"))
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    data = nc.load_conll()
    tr_s, tr_t = [r["tokens"] for r in data["train"]], [r["tags"] for r in data["train"]]
    va_s, va_t = [r["tokens"] for r in data["validation"]], [r["tags"] for r in data["validation"]]
    te_s, te_t = [r["tokens"] for r in data["test"]], [r["tags"] for r in data["test"]]
    sweep, best, best_model = [], None, None
    for C in C_GRID:
        t0 = time.perf_counter()
        m = BaselineNER(C).fit(tr_s, tr_t)
        fit_s = time.perf_counter() - t0
        pv, _ = m.predict(va_s)
        vf = nc.entity_metrics(va_t, pv)["micro"]["f1"]
        sweep.append({"C": C, "val_entity_f1": vf, "fit_s": fit_s, "n_features": m.n_features})
        if verbose:
            print(f"C={C}: val entity F1={vf:.4f} fit={fit_s:.0f}s features={m.n_features}", flush=True)
        if best is None or vf > best:
            best, best_model = vf, m
    pv, _ = best_model.predict(va_s)
    pt, ct = best_model.predict(te_s)
    result = {"name": "baseline", "config": {"features": "window+-2 word/shape/affix/bigram", "tfidf": "sublinear", "clf": "LogisticRegression lbfgs",
                                             "C": best_model.C, "selected_by": "val entity F1"},
              "n_features": best_model.n_features, "sweep": sweep,
              "val": {"entity": nc.entity_metrics(va_t, pv), "token": nc.token_metrics(va_t, pv)},
              "test": {"entity": nc.entity_metrics(te_t, pt), "token": nc.token_metrics(te_t, pt)},
              "invalid_bio_transitions_test": sum(nc.invalid_bio_transitions(t) for t in pt),
              "invalid_bio_transitions_gold_test": sum(nc.invalid_bio_transitions(t) for t in te_t)}
    (BASELINE_DIR / "preds_test.json").write_text(json.dumps({"pred_tags": pt, "confidence": ct}), encoding="utf8")
    (BASELINE_DIR / "result.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    joblib.dump(best_model, BASELINE_DIR / "model.joblib", compress=3)
    return result


def load_baseline() -> BaselineNER:
    return joblib.load(BASELINE_DIR / "model.joblib")


if __name__ == "__main__":
    import baseline_ner as _m            # luu model.joblib voi class thuoc module `baseline_ner` (khong phai `__main__`)
    r = _m.run_baseline()
    e = r["test"]["entity"]["micro"]
    print(f"TEST entity P={e['precision']:.4f} R={e['recall']:.4f} F1={e['f1']:.4f}  token acc={r['test']['token']['accuracy']:.4f}")
