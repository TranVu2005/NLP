"""LAB 10 - Baseline (muc 21): keyword overlap -> candidate sentence -> candidate span. Khong hoc tham so tu nhan (khong dung gold answer).

    Question --(content words, IDF)--> chon cau trong context co overlap cao nhat
             --(kieu cau hoi: when/who/where/how many/...)--> chon span ung vien trong cau do
                      (cum so/ngay, cum chu viet hoa, hoac cum tu noi dung) gan nhat voi tu khoa trung
Hai bien the:
    sentence_only : tra ve ca cau tot nhat (do chat luong buoc 'tim cau')
    span          : baseline day du (cau + span)
Muc dich (de): xac dinh Transformer co thuc su tan dung tuong tac question/context tot hon mot heuristic don gian hay khong.

CLI:  python baseline_qa.py            # danh gia val + test (SQuAD dev) va ghi runs/baseline/
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # noqa: F401,I001  (import pandas truoc torch tren Windows - qa_model import torch)

import qa_model as qm

HERE = Path(__file__).resolve().parent

STOPWORDS = set("""a an the of in on at to for from by with about as into like through after over between out against during without
before under around among and or but if while because than so that this these those it its is are was were be been being am do does did
have has had having not no nor can could may might shall should will would must also only very just more most some any each other such
there here their they them he she him her his hers we us our you your i me my what which who whom whose when where why how whom s t""".split())
WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
NUMBER_UNITS = {"million", "billion", "thousand", "hundred", "percent", "%", "bc", "ad", "bce", "ce", "century", "centuries", "years", "year"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
CAP_CONNECT = {"of", "de", "von", "van", "la", "del", "the", "and", "for", "st"}

SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
TOKEN_RE = re.compile(r"\w+(?:[.,'’\-/]\w+)*|[^\s\w]")


def split_sentences(context: str) -> list[tuple[int, int]]:
    """Cac khoang ky tu (start, end) cua tung cau (regex: sau . ! ? roi khoang trang va chu hoa/chu so)."""
    spans, last = [], 0
    for m in SENT_SPLIT.finditer(context):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(context)))
    return [(a, b) for a, b in spans if b > a]


def word_tokens(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start() + offset, m.end() + offset) for m in TOKEN_RE.finditer(text)]


def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content_stems(text: str) -> list[str]:
    return [stem(t) for t, _, _ in word_tokens(text) if re.match(r"\w", t) and t.lower() not in STOPWORDS]


def question_type(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(when|what year|which year|in what year|what century|what date|what time|what decade|what month|which century)\b", ql):
        return "date"
    if re.search(r"\b(how many|how much|how long|how old|how far|how large|how big|how tall|how high|how often|what percent|what percentage|what number)\b", ql):
        return "number"
    if re.search(r"\b(who|whom|whose)\b", ql):
        return "person"
    if re.search(r"\b(where|what city|what country|what state|which city|which country|what river|what region|what place|what town)\b", ql):
        return "place"
    return "other"


class KeywordBaseline:
    """fit(contexts): chi dem document frequency (cap cau) cho IDF - khong dung nhan. predict(question, context) -> dict."""

    def __init__(self) -> None:
        self.df: collections.Counter = collections.Counter()
        self.n_sent = 0

    def fit(self, contexts: Sequence[str]) -> "KeywordBaseline":
        for c in set(contexts):
            for a, b in split_sentences(c):
                self.df.update(set(content_stems(c[a:b])))
                self.n_sent += 1
        return self

    def idf(self, w: str) -> float:
        return math.log((self.n_sent + 1) / (self.df.get(w, 0) + 1)) + 1.0

    # ---- buoc 1: chon cau ------------------------------------------------- #
    def rank_sentences(self, question: str, context: str) -> list[tuple[float, tuple[int, int], set[str]]]:
        q = set(content_stems(question))
        out = []
        for a, b in split_sentences(context):
            s = set(content_stems(context[a:b]))
            matched = q & s
            out.append((sum(self.idf(w) for w in matched), (a, b), matched))
        out.sort(key=lambda x: (-x[0], x[1][0]))           # diem giam dan; hoa -> cau dung truoc
        return out

    # ---- buoc 2: chon span trong cau --------------------------------------- #
    @staticmethod
    def candidates(sent_tokens: list[tuple[str, int, int]], qstems: set[str]) -> list[dict[str, Any]]:
        """Cum ung vien: so/ngay, chuoi chu viet hoa (cho phep noi bang of/de/the...), cum tu noi dung. Bo cum chua tu cua cau hoi."""
        n = len(sent_tokens)
        low = [t.lower() for t, _, _ in sent_tokens]
        is_word = [bool(re.match(r"\w", t)) for t, _, _ in sent_tokens]
        is_num = [any(ch.isdigit() for ch in t) or low[i] in NUMBER_UNITS or low[i] in MONTHS for i, (t, _, _) in enumerate(sent_tokens)]
        is_cap = [is_word[i] and t[0].isupper() and low[i] not in STOPWORDS and not (i == 0 and low[i] in WH_WORDS)
                  for i, (t, _, _) in enumerate(sent_tokens)]
        is_content = [is_word[i] and low[i] not in STOPWORDS for i in range(n)]
        cands: list[dict[str, Any]] = []

        def add(kind: str, i: int, j: int) -> None:          # token i..j-1
            toks = sent_tokens[i:j]
            if not toks:
                return
            if any(stem(t.lower()) in qstems for t, _, _ in toks if re.match(r"\w", t)):
                return
            cands.append({"kind": kind, "i": i, "j": j, "start": toks[0][1], "end": toks[-1][2]})

        def runs(flag: list[bool], kind: str, connect: set[str] | None = None) -> None:
            i = 0
            while i < n:
                if not flag[i]:
                    i += 1
                    continue
                j = i + 1
                while j < n:
                    if flag[j]:
                        j += 1
                    elif connect and low[j] in connect and j + 1 < n and flag[j + 1]:
                        j += 2
                    else:
                        break
                add(kind, i, j)
                i = j

        runs(is_num, "number")
        runs(is_cap, "cap", CAP_CONNECT)
        i = 0
        while i < n:                                          # cum noi dung toi da 6 token
            if not is_content[i]:
                i += 1
                continue
            j = i
            while j < n and is_content[j] and j - i < 6:
                j += 1
            add("content", i, j)
            i = j
        return cands

    def predict(self, question: str, context: str, mode: str = "span") -> dict[str, Any]:
        ranked = self.rank_sentences(question, context)
        best_score, (sa, sb), _ = ranked[0]
        if mode == "sentence_only":
            return {"text": context[sa:sb], "start_char": sa, "end_char": sb, "sentence": (sa, sb), "sent_score": best_score}
        qtype = question_type(question)
        qstems = set(content_stems(question))
        prefer = {"date": ["number"], "number": ["number"], "person": ["cap"], "place": ["cap"], "other": ["cap", "number", "content"]}[qtype]
        for rank, (score, (a, b), matched) in enumerate(ranked[:3]):
            toks = word_tokens(context[a:b], a)
            matched_pos = [i for i, (t, _, _) in enumerate(toks) if re.match(r"\w", t) and stem(t.lower()) in qstems]
            cands = self.candidates(toks, qstems)
            for kinds in ([prefer[0]] if qtype in ("date", "number") else prefer, ["cap", "number", "content"]):
                pool = [c for c in cands if c["kind"] in kinds]
                if qtype == "date":
                    pool = [c for c in pool if any(ch.isdigit() for ch in context[c["start"]:c["end"]]) or
                            any(m in context[c["start"]:c["end"]].lower() for m in MONTHS)] or pool
                if pool:
                    def dist(c):
                        return min((min(abs(c["i"] - m), abs(c["j"] - 1 - m)) for m in matched_pos), default=0)
                    c = min(pool, key=lambda c: (dist(c), c["i"]))
                    return {"text": context[c["start"]:c["end"]], "start_char": c["start"], "end_char": c["end"],
                            "sentence": (sa, sb), "sent_score": best_score, "kind": c["kind"], "qtype": qtype}
        a_, b_ = ranked[0][1]
        toks = word_tokens(context[a_:b_], a_)[:3]
        s, e = (toks[0][1], toks[-1][2]) if toks else (a_, b_)
        return {"text": context[s:e], "start_char": s, "end_char": e, "sentence": (a_, b_), "sent_score": best_score, "kind": "fallback", "qtype": qtype}


def evaluate_baseline(bl: KeywordBaseline, split: Sequence[dict[str, Any]], mode: str = "span") -> dict[str, Any]:
    preds = [bl.predict(r["question"], r["context"], mode) for r in split]
    golds = [r["answers"]["text"] for r in split]
    m = qm.evaluate_predictions([p["text"] for p in preds], golds)
    hit = []
    for p, r in zip(preds, split):
        a, b = p["sentence"]
        hit.append(any(a <= st < b for st in r["answers"]["answer_start"]))
    m["sentence_hit"] = sum(hit) / len(hit)                     # answer (bat dau) nam trong cau duoc chon
    return {"metrics": m, "preds": preds}


def main() -> None:
    data = qm.make_splits()
    t0 = time.perf_counter()
    bl = KeywordBaseline().fit([r["context"] for r in data["train"]])
    out = {"name": "baseline", "fit_time_s": time.perf_counter() - t0}
    rdir = qm.RUNS_DIR / "baseline"
    rdir.mkdir(parents=True, exist_ok=True)
    for mode in ("sentence_only", "span"):
        for split in ("validation", "test"):
            t1 = time.perf_counter()
            r = evaluate_baseline(bl, data[split], mode)
            r["metrics"]["time_s"] = time.perf_counter() - t1
            out[f"{mode}_{split}"] = r["metrics"]
            print(mode, split, {k: round(v, 4) for k, v in r["metrics"].items()}, flush=True)
            if split == "test":
                (rdir / f"preds_test_{mode}.json").write_text(json.dumps(
                    {"ids": [x["id"] for x in data["test"]],
                     "preds": [{k: p[k] for k in ("text", "start_char", "end_char", "sentence")} for p in r["preds"]]}), encoding="utf8")
    (rdir / "result.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
