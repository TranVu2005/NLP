"""LAB 10 - Document QA (muc 27): mini QA system  question -> (document da cho san) -> QA model -> answer span.

    - KHONG phai RAG: document/context duoc cung cap truc tiep (LAB 11-12 moi bo sung buoc retrieval).
    - Dung model fine-tune cua LAB 10 (qa_model.QAModel, checkpoint `runs/<MAIN_RUN>/model.pt`) voi sliding window
      (document dai hon max_length duoc chia window chong nhau) - KHONG dung pipeline("question-answering").
    - Voi moi cau hoi: tra ve span, vi tri ky tu, diem (start+end logit), "confidence" (softmax tren cac span ung vien),
      so window da dung, cau chua dap an. Co the bo qua cau tra loi khi (score - null_score) < nguong (--null_margin).

CLI:
    python document_qa.py --demo                                              # 2 tai lieu mau (Wikipedia-style + FAQ)
    python document_qa.py --context_file doc.txt --question "Who wrote it?" --question "When?"
    python document_qa.py --context "Paris is the capital of France." --question "What is the capital of France?"
    python document_qa.py --context_file doc.txt --interactive
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # noqa: F401,I001  (pandas truoc torch tren Windows)
import numpy as np
import torch

import qa_model as qm

DEMO_DOCS: dict[str, dict[str, Any]] = {
    "transformer_wiki": {
        "title": "Wikipedia-style paragraph: Transformer and BERT",
        "context": (
            "The Transformer is a neural network architecture introduced in 2017 in the paper \"Attention Is All You Need\" by researchers "
            "at Google. It replaces recurrence with self-attention, which lets every token in a sequence attend directly to every other token. "
            "BERT, short for Bidirectional Encoder Representations from Transformers, was released by Google in 2018. BERT uses only the "
            "encoder part of the Transformer and is pretrained with two objectives: masked language modeling and next sentence prediction. "
            "The base version of BERT has 12 layers, a hidden size of 768 and 12 attention heads, giving about 110 million parameters. "
            "After pretraining, BERT can be fine-tuned for tasks such as text classification, named entity recognition and question answering "
            "by adding a small prediction head on top of its contextual representations."),
        "questions": [
            "When was the Transformer introduced?",
            "Who introduced the Transformer?",
            "What does the Transformer use instead of recurrence?",
            "What are the two pretraining objectives of BERT?",
            "How many layers does BERT base have?",
            "How many parameters does BERT base have?",
            "What can BERT be fine-tuned for?",
            "What is the capital of France?",          # khong co trong tai lieu -> extractive QA van tra ve mot span (xem muc 26 cua de)
        ],
    },
    "refund_faq": {
        "title": "FAQ: refund policy of a fictional online shop",
        "context": (
            "Customers may return unused items within 30 days of delivery for a full refund. To start a return, open the order page and "
            "choose the option called Request a return. Refunds are issued to the original payment method within 5 to 7 business days after "
            "the returned item reaches our warehouse. Shipping fees are not refunded unless the item arrived damaged or the wrong item was sent. "
            "Gift cards and downloadable software cannot be returned. Customer support is available Monday to Friday from 9 a.m. to 6 p.m. "
            "by email at support@example.com or by phone at 555-0142."),
        "questions": [
            "How many days do customers have to return an item?",
            "How long does a refund take?",
            "Which items cannot be returned?",
            "When are shipping fees refunded?",
            "What are the support hours?",
            "How can I contact customer support?",
            "Do you ship to Canada?",                  # khong co trong tai lieu
        ],
    },
}


class DocumentQA:
    """Nap model da train mot lan; answer(question, context) / answer_many(questions, context)."""

    def __init__(self, run: str = qm.MAIN_RUN, device: torch.device | None = None, max_len: int | None = None,
                 stride: int | None = None, null_margin: float | None = None, max_answer_len: int = qm.MAX_ANSWER_LEN):
        self.device = device or qm.get_device()
        self.model, self.tok, self.cfg = qm.load_trained(run, self.device)
        self.max_len = max_len or self.cfg["max_len"]
        self.stride = stride or qm.default_stride(self.max_len)
        self.null_margin = null_margin
        self.max_answer_len = max_answer_len

    @torch.no_grad()
    def answer_many(self, questions: Sequence[str], context: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Tra loi nhieu cau hoi tren CUNG mot document. Moi muc: text, start_char, end_char, score, confidence, null_score,
        margin = score - null_score, n_windows, sentence, candidates (top_k)."""
        ctxs = [context] * len(questions)
        feats = qm.QAFeatures(self.tok, list(questions), ctxs, self.max_len, self.stride)
        fw = qm.forward_features(self.model, feats, self.tok.pad_token_id, self.device)
        by_ex: dict[int, list[int]] = {}
        for i, f in enumerate(feats.feats):
            by_ex.setdefault(f["example_idx"], []).append(i)
        out = []
        for ex, q in enumerate(questions):
            cands: dict[tuple[int, int], float] = {}
            null = float("inf")
            for fi in by_ex[ex]:
                s, e, offs = fw["start"][fi], fw["end"][fi], feats.feats[fi]["offsets"]
                null = min(null, float(s[0] + e[0]))
                for a in np.argsort(s)[::-1][:qm.N_BEST]:
                    if offs[a] is None:
                        continue
                    for b in np.argsort(e)[::-1][:qm.N_BEST]:
                        if offs[b] is None or b < a or b - a + 1 > self.max_answer_len:
                            continue
                        key = (offs[a][0], offs[b][1])
                        sc = float(s[a] + e[b])
                        if key not in cands or sc > cands[key]:
                            cands[key] = sc
            ranked = sorted(cands.items(), key=lambda kv: -kv[1])
            scores = np.array([v for _, v in ranked])
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            (a, b), best = ranked[0]
            sent = _sentence_of(context, a)
            res = {"question": q, "text": context[a:b], "start_char": a, "end_char": b, "score": best, "null_score": null,
                   "margin": best - null, "confidence": float(probs[0]), "n_windows": len(by_ex[ex]), "sentence": sent,
                   "candidates": [(context[x:y], float(p)) for ((x, y), _), p in zip(ranked[:top_k], probs[:top_k])]}
            res["abstain"] = self.null_margin is not None and res["margin"] < self.null_margin
            out.append(res)
        return out

    def answer(self, question: str, context: str, top_k: int = 3) -> dict[str, Any]:
        return self.answer_many([question], context, top_k)[0]


def _sentence_of(context: str, pos: int) -> str:
    import baseline_qa as bq
    for a, b in bq.split_sentences(context):
        if a <= pos < b:
            return context[a:b]
    return context[max(0, pos - 80):pos + 80]


def highlight(context: str, a: int, b: int) -> str:
    return context[:a] + "[[" + context[a:b] + "]]" + context[b:]


def format_result(r: dict[str, Any]) -> str:
    ans = "(khong tra loi: margin < nguong)" if r["abstain"] else repr(r["text"])
    cands = "; ".join(f"{t!r} {p:.2f}" for t, p in r["candidates"])
    return (f"Q: {r['question']}\n   A: {ans}  [chars {r['start_char']}-{r['end_char']}, score {r['score']:.2f}, margin(score-null) {r['margin']:+.2f}, "
            f"confidence {r['confidence']:.2f}, windows {r['n_windows']}]\n   top-k: {cands}\n   sentence: {r['sentence']}")


def run_demo(dqa: DocumentQA) -> list[dict[str, Any]]:
    allres = []
    for key, doc in DEMO_DOCS.items():
        print("=" * 100, f"\n{doc['title']}\n{doc['context']}\n")
        for r in dqa.answer_many(doc["questions"], doc["context"]):
            print(format_result(r))
            allres.append({"doc": key, **r})
    return allres


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=qm.MAIN_RUN, help="ten thu muc trong runs/ (can co model.pt)")
    ap.add_argument("--context")
    ap.add_argument("--context_file")
    ap.add_argument("--question", action="append", default=[])
    ap.add_argument("--max_len", type=int)
    ap.add_argument("--null_margin", type=float, default=None, help="neu score - null_score < nguong thi tu choi tra loi")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    dqa = DocumentQA(a.run, max_len=a.max_len, null_margin=a.null_margin)
    if a.demo:
        run_demo(dqa)
        return
    context = a.context or (Path(a.context_file).read_text(encoding="utf8") if a.context_file else None)
    if context is None:
        sys.exit("can --context hoac --context_file (hoac --demo)")
    for q in a.question:
        print(format_result(dqa.answer(q, context)))
    if a.interactive:
        while True:
            q = input("question> ").strip()
            if not q:
                break
            r = dqa.answer(q, context)
            print(format_result(r))
            print("   in context:", highlight(context, r["start_char"], r["end_char"])[:400])


if __name__ == "__main__":
    main(sys.argv[1:])
