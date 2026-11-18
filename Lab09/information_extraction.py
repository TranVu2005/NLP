"""LAB 09 - Information Extraction (muc 25-27), Option A (News): PERSON / ORG / LOCATION.

    raw text -> tach cau/tu (regex) -> NER (Transformer hoac baseline) -> gop B-/I- thanh entity
             -> {"PERSON": [...], "ORG": [...], "LOCATION": [...]}

`MISC` bi bo (Option A chi trich xuat 3 loai). Danh gia entity-level tren 30 cau tu gan nhan tay (`ie_gold.json`):
mot entity dung khi KHOP CA khoang ky tu (ranh gioi) LAN loai.

CLI:
    python information_extraction.py --text "Barack Obama visited Paris."
    python information_extraction.py --evaluate          # P/R/F1 tren ie_gold.json (Transformer + baseline)
Can runs/<MAIN_RUN>/model.pt (python ner_core.py --run_grid) cho Transformer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import ner_core as nc

GOLD_PATH = Path(__file__).resolve().parent / "ie_gold.json"
TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}            # MISC -> bo
IE_TYPES = ["PERSON", "ORG", "LOCATION"]
# Theo quy uoc token cua CoNLL: giu chu viet tat co dau cham ("U.S.", "Mr.", "Dr."), tach "'s" va dau cau rieng
WORD_RE = re.compile(r"(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr)\.(?=\s)|(?:[A-Za-z]\.){2,}|\d+(?:[-.,]\d+)*|\w+(?:-\w+)*|['’]s\b|[^\w\s]")

Tagger = Callable[[Sequence[Sequence[str]]], list[list[str]]]


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def extract_entities(text: str, tagger: Tagger) -> list[dict[str, Any]]:
    """Tra ve danh sach {type, text, start, end} (offset ky tu trong `text`), da bo MISC."""
    toks = tokenize_with_offsets(text)
    tags = tagger([[t[0] for t in toks]])[0]
    out = []
    for typ, s, e in nc.tags_to_entities(tags):
        if typ not in TYPE_MAP:
            continue
        a, b = toks[s][1], toks[e - 1][2]
        out.append({"type": TYPE_MAP[typ], "text": text[a:b], "start": a, "end": b})
    return out


def to_structured(entities: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Bieu dien co cau truc: {"PERSON": ["Barack Obama"], "LOCATION": ["Paris"]} (khong trung, giu thu tu xuat hien)."""
    out: dict[str, list[str]] = {}
    for e in entities:
        lst = out.setdefault(e["type"], [])
        if e["text"] not in lst:
            lst.append(e["text"])
    return out


def transformer_tagger(run: str = nc.MAIN_RUN) -> Tagger:
    import torch
    model, tok, cfg = nc.load_trained(run)
    device = nc.get_device()

    def tag(sentences: Sequence[Sequence[str]]) -> list[list[str]]:
        ds = nc.NERDataset([{"id": str(i), "tokens": list(s), "tags": ["O"] * len(s)} for i, s in enumerate(sentences)], tok, "A", cfg["max_len"])
        return nc.predict_dataset(model, ds, tok.pad_token_id, device)["pred_tags"]
    return tag


def baseline_tagger() -> Tagger:
    from baseline_ner import load_baseline
    m = load_baseline()
    return lambda sentences: m.predict(sentences)[0]


def load_gold(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    """Doc gold: moi entity -> khoang ky tu (tim chuoi con trong cau; bo qua cac lan xuat hien chong lap khoang da gan,
    vd. "Jordan" trong "Michael Jordan" khi gold co ca `Michael Jordan` va `Jordan`)."""
    sents = json.loads(path.read_text(encoding="utf8"))["sentences"]
    out = []
    for s in sents:
        used: list[tuple[int, int]] = []
        ents = []
        for typ, surface in s["entities"]:
            for m in re.finditer(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", s["text"]):
                if not any(m.start() < b and a < m.end() for a, b in used):
                    used.append((m.start(), m.end()))
                    ents.append((typ, m.start(), m.end()))
                    break
            else:
                raise ValueError(f"gold entity khong tim thay: {surface!r} trong {s['text']!r}")
        out.append({"text": s["text"], "gold": ents})
    return out


def evaluate_ie(tagger: Tagger, gold: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    gold = gold if gold is not None else load_gold()
    per_type = {t: [0, 0, 0] for t in IE_TYPES}                          # tp, fp, fn
    errors = []
    for i, g in enumerate(gold):
        pred = {(e["type"], e["start"], e["end"]) for e in extract_entities(g["text"], tagger)}
        ref = set(g["gold"])
        for t in IE_TYPES:
            pt, rt = {x for x in pred if x[0] == t}, {x for x in ref if x[0] == t}
            per_type[t][0] += len(pt & rt)
            per_type[t][1] += len(pt - rt)
            per_type[t][2] += len(rt - pt)
        if pred != ref:
            errors.append({"sent": i, "text": g["text"], "missing": sorted(ref - pred), "spurious": sorted(pred - ref)})
    res = {t: nc._prf(*v) for t, v in per_type.items()}
    tp, fp, fn = (sum(v[k] for v in per_type.values()) for k in range(3))
    res["micro"] = nc._prf(tp, fp, fn)
    return {"metrics": res, "errors": errors, "n_sentences": len(gold), "n_gold_entities": tp + fn}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="van ban can trich xuat")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--model", choices=["transformer", "baseline"], default="transformer")
    a = ap.parse_args(argv)
    tagger = transformer_tagger() if a.model == "transformer" else baseline_tagger()
    if a.text:
        print(json.dumps(to_structured(extract_entities(a.text, tagger)), indent=2, ensure_ascii=False))
    if a.evaluate:
        r = evaluate_ie(tagger)
        for k, v in r["metrics"].items():
            print(f"{k:9s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (tp={v['tp']} fp={v['fp']} fn={v['fn']})")


if __name__ == "__main__":
    main(sys.argv[1:])
