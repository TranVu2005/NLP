"""LAB 10 - Loi dung chung cho Extractive QA tren SQuAD v1.1: du lieu, tokenize + offset mapping, char->token span
alignment, QAModel (AutoModel + head start/end tu viet), QA loss, postprocess span, EM/F1 (tu cai) va vong train/eval.

Pipeline (muc 15-16 cua de):
    (question, context) -> tokenizer(truncation="only_second", max_length, stride, return_overflowing_tokens,
                                     return_offsets_mapping) -> input_ids / attention_mask / (token_type_ids)
        -> char span (answer_start, answer_end) --offset_mapping--> (start_token, end_token)   [per window]
        -> pretrained encoder -> H [B,T,d] -> Linear(d,2) -> start_logits, end_logits [B,T]
        -> L = (CE(start) + CE(end)) / 2 -> argmax / n-best -> char span -> answer text

`QAModel` tu viet (AutoModel + head tu viet), KHONG dung AutoModelForQuestionAnswering va KHONG dung pipeline("question-answering").
EM/F1 tu cai theo script SQuAD v1.1 chinh thuc (normalize: lowercase, bo dau cau, bo a/an/the, gom khoang trang).

CLI:
    python qa_model.py --max_len 384 --seed 42      # 1 run (cache vao runs/)
    python qa_model.py --run_grid                   # toan bo run cua bao cao

LUU Y Windows: phai import pandas TRUOC torch (neu khong process co the crash).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # noqa: F401,I001  (bat buoc import truoc torch tren Windows)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ["LAB10_RUNS_DIR"]) if os.environ.get("LAB10_RUNS_DIR") else HERE / "runs"   # (chi dung de chay thu notebook)
DATA_CACHE = HERE / "data_cache"

SEED = 42
MODEL_NAME = "bert-base-cased"
DATASET_ID = "rajpurkar/squad"
N_TRAIN = 20000          # so cau hoi train (tap con cua SQuAD train 87,599 - gioi han GPU 6GB / thoi gian)
N_VAL = 1000             # validation: cau hoi tu cac bai (title) KHONG nam trong train subset
MAX_ANSWER_LEN = 30      # so token toi da cua span du doan
N_BEST = 20
NEG = -1e4


# --------------------------------------------------------------------------- #
# Reproducibility / env
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_info() -> dict[str, Any]:
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.cuda.is_available(), "platform": platform.platform()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


# --------------------------------------------------------------------------- #
# Data: SQuAD v1.1
# --------------------------------------------------------------------------- #
def _flatten(r: dict[str, Any]) -> dict[str, Any]:
    return {"id": r["id"], "title": r["title"], "context": r["context"], "question": r["question"].strip(),
            "answers": {"text": list(r["answers"]["text"]), "answer_start": [int(x) for x in r["answers"]["answer_start"]]}}


def load_squad_raw(use_cache: bool = True) -> dict[str, list[dict[str, Any]]]:
    """{'train' (87,599), 'dev' (10,570)} - dev chinh thuc (>= 3 gold answer/cau) dung lam TEST. Cache jsonl."""
    DATA_CACHE.mkdir(exist_ok=True)
    files = {"train": DATA_CACHE / "squad_train_full.jsonl", "dev": DATA_CACHE / "squad_dev_full.jsonl"}
    if use_cache and all(f.exists() for f in files.values()):
        return {k: [json.loads(line) for line in f.read_text(encoding="utf8").splitlines()] for k, f in files.items()}
    import datasets
    ds = datasets.load_dataset(DATASET_ID)
    out = {"train": [_flatten(r) for r in ds["train"]], "dev": [_flatten(r) for r in ds["validation"]]}
    for k, f in files.items():
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out[k]), encoding="utf8")
    return out


def make_splits(n_train: int = N_TRAIN, n_val: int = N_VAL, seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    """train (n_train cau, mau ngau nhien) / validation (n_val cau, tu cac bai KHONG co trong train subset) / test (= dev chinh thuc).
    Chia theo `title` (bai Wikipedia) de validation khong ro ri context voi train. Cache de tai lap."""
    f = DATA_CACHE / f"splits_tr{n_train}_va{n_val}_s{seed}.json"
    raw = load_squad_raw()
    if f.exists():
        ids = json.loads(f.read_text(encoding="utf8"))
        byid = {r["id"]: r for r in raw["train"]}
        return {"train": [byid[i] for i in ids["train"]], "validation": [byid[i] for i in ids["validation"]], "test": raw["dev"]}
    rng = random.Random(seed)
    titles = sorted({r["title"] for r in raw["train"]})
    rng.shuffle(titles)
    n_val_titles = max(1, int(round(len(titles) * 0.10)))
    val_titles = set(titles[:n_val_titles])
    val_pool = [r for r in raw["train"] if r["title"] in val_titles]
    train_pool = [r for r in raw["train"] if r["title"] not in val_titles]
    val = rng.sample(val_pool, min(n_val, len(val_pool)))
    train = rng.sample(train_pool, min(n_train, len(train_pool)))
    DATA_CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps({"train": [r["id"] for r in train], "validation": [r["id"] for r in val]}), encoding="utf8")
    return {"train": train, "validation": val, "test": raw["dev"]}


# --------------------------------------------------------------------------- #
# EM / F1 (SQuAD v1.1 official semantics, tu cai)
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """lowercase -> bo dau cau -> bo mao tu a/an/the -> gom khoang trang."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_prf(pred: str, gold: str) -> tuple[float, float, float]:
    """(precision, recall, F1) muc token cua prediction va gold (sau normalize); overlap = giao multiset."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        same = float(p == g)          # ca hai rong -> 1, mot ben rong -> 0
        return same, same, same
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0, 0.0, 0.0
    prec, rec = n / len(p), n / len(g)
    return prec, rec, 2 * prec * rec / (prec + rec)


def f1_score(pred: str, gold: str) -> float:
    return f1_prf(pred, gold)[2]


def metric_max_over_golds(fn, pred: str, golds: Sequence[str]) -> float:
    return max(fn(pred, g) for g in golds)


def evaluate_predictions(preds: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, float]:
    """EM, F1 trung binh (0..1); moi cau lay max tren cac gold answer."""
    em = [metric_max_over_golds(exact_match, p, g) for p, g in zip(preds, golds)]
    f1 = [metric_max_over_golds(f1_score, p, g) for p, g in zip(preds, golds)]
    return {"em": float(np.mean(em)), "f1": float(np.mean(f1)), "n": len(em)}


def per_example_scores(preds: Sequence[str], golds: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray]:
    em = np.array([metric_max_over_golds(exact_match, p, g) for p, g in zip(preds, golds)])
    f1 = np.array([metric_max_over_golds(f1_score, p, g) for p, g in zip(preds, golds)])
    return em, f1


# --------------------------------------------------------------------------- #
# Tokenize (question, context) + offset mapping + char->token span alignment
# --------------------------------------------------------------------------- #
def tokenize_qa(tokenizer, questions: Sequence[str], contexts: Sequence[str], max_len: int, stride: int):
    """Dung dung call cua de (muc 15): truncation='only_second' (chi cat context), max_length, stride (overlap giua cac window),
    return_overflowing_tokens + return_offsets_mapping."""
    return tokenizer([q.strip() for q in questions], list(contexts), truncation="only_second", max_length=max_len,
                     stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True)


def context_token_range(sequence_ids: Sequence[int | None]) -> tuple[int, int]:
    """(first, last) index (bao gom) cac token thuoc context (sequence id = 1)."""
    idx = [i for i, s in enumerate(sequence_ids) if s == 1]
    return idx[0], idx[-1]


def align_span(offsets: Sequence[tuple[int, int]], sequence_ids: Sequence[int | None], answer_start: int,
               answer_end: int) -> tuple[int, int] | None:
    """Anh xa character span [answer_start, answer_end) (answer_end la ranh gioi phai, EXCLUSIVE) sang token span (start_tok, end_tok)
    trong MOT window. Tra ve None neu answer KHONG nam tron trong context cua window (nam ngoai window / bi truncation cat).

    Logic cua de (muc 16): start_tok la token co `start <= answer_start < end`; end_tok la token co `start < answer_end <= end`.
    Cai dat: duyet chi cac token context; neu answer bat dau/ket thuc trong khoang trong giua hai token (khong token nao chua ky tu
    do) thi lay token ke can (start: token dau tien co end > answer_start; end: token cuoi cung co start < answer_end)."""
    cs, ce = context_token_range(sequence_ids)
    if offsets[cs][0] > answer_start or offsets[ce][1] < answer_end:
        return None
    start_tok = end_tok = None
    for i in range(cs, ce + 1):
        s, e = offsets[i]
        if e <= s:                       # token rong (khong co ky tu) - bo qua
            continue
        if start_tok is None and e > answer_start:
            start_tok = i                # token dau tien ket thuc SAU answer_start  (= start <= answer_start < end khi co chua)
        if s < answer_end:
            end_tok = i                  # token cuoi cung bat dau TRUOC answer_end (= start < answer_end <= end khi co chua)
    if start_tok is None or end_tok is None or end_tok < start_tok:
        return None
    return start_tok, end_tok


def alignment_kind(offsets, sequence_ids, answer_start: int, answer_end: int) -> str:
    """Phan loai tung truong hop cua de: 'outside_window' (nam ngoai/bi cat), 'exact', 'start_mid_token', 'end_mid_token', 'both_mid_token'."""
    sp = align_span(offsets, sequence_ids, answer_start, answer_end)
    if sp is None:
        return "outside_window"
    s_mid = offsets[sp[0]][0] != answer_start
    e_mid = offsets[sp[1]][1] != answer_end
    return {(False, False): "exact", (True, False): "start_mid_token", (False, True): "end_mid_token", (True, True): "both_mid_token"}[(s_mid, e_mid)]


class QAFeatures:
    """Cac feature (window) cua mot tap (question, context). Moi feature: input_ids, attention_mask, (token_type_ids),
    offset_mapping (chi context; None cho question/special), example_idx, start/end positions (train: None -> (0,0)=[CLS])."""

    def __init__(self, tokenizer, questions: Sequence[str], contexts: Sequence[str], max_len: int, stride: int,
                 answers: Sequence[tuple[int, int]] | None = None):
        enc = tokenize_qa(tokenizer, questions, contexts, max_len, stride)
        self.max_len, self.stride = max_len, stride
        self.n_examples = len(questions)
        self.feats: list[dict[str, Any]] = []
        self.n_no_answer_windows = 0
        self.n_answer_cut = 0                      # example co answer nhung KHONG window nao chua tron answer
        had_window = collections.defaultdict(bool)
        self.has_token_type = "token_type_ids" in enc
        for i in range(len(enc["input_ids"])):
            ex = enc["overflow_to_sample_mapping"][i]
            seq = enc.sequence_ids(i)
            offs = [tuple(o) if seq[k] == 1 else None for k, o in enumerate(enc["offset_mapping"][i])]
            f = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "example_idx": ex,
                 "offsets": offs, "sequence_ids": seq}
            if self.has_token_type:
                f["token_type_ids"] = enc["token_type_ids"][i]
            if answers is not None:
                a_s, a_e = answers[ex]
                sp = align_span([o if o is not None else (0, 0) for o in offs], seq, a_s, a_e)
                if sp is None:
                    f["start_position"], f["end_position"] = 0, 0           # [CLS]: answer khong nam trong window nay
                    self.n_no_answer_windows += 1
                else:
                    f["start_position"], f["end_position"] = sp
                    had_window[ex] = True
            self.feats.append(f)
        if answers is not None:
            self.n_answer_cut = sum(1 for e in range(self.n_examples) if not had_window[e])

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {**self.feats[i], "idx": i}

    def stats(self) -> dict[str, Any]:
        lens = [len(f["input_ids"]) for f in self.feats]
        return {"n_examples": self.n_examples, "n_features": len(self.feats), "windows_per_example": len(self.feats) / self.n_examples,
                "n_no_answer_windows": self.n_no_answer_windows, "n_examples_answer_lost": self.n_answer_cut,
                "mean_len": float(np.mean(lens)), "max_len_seen": int(max(lens))}


def make_collate(pad_id: int, with_token_type: bool):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        T = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        tt = torch.zeros((len(batch), T), dtype=torch.long)
        sp = torch.zeros(len(batch), dtype=torch.long)
        ep = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
            if with_token_type:
                tt[i, :n] = torch.tensor(b["token_type_ids"])
            sp[i] = b.get("start_position", 0)
            ep[i] = b.get("end_position", 0)
        out = {"input_ids": ids, "attention_mask": mask, "start_positions": sp, "end_positions": ep, "idx": [b["idx"] for b in batch]}
        if with_token_type:
            out["token_type_ids"] = tt
        return out
    return collate


def make_loader(feats: QAFeatures, pad_id: int, batch_size: int, shuffle: bool, seed: int = SEED) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(feats, batch_size=batch_size, shuffle=shuffle, generator=g, collate_fn=make_collate(pad_id, feats.has_token_type))


# --------------------------------------------------------------------------- #
# Model + loss
# --------------------------------------------------------------------------- #
class QAModel(nn.Module):
    """H = encoder(x) in R^{B x T x d};  [start_logits, end_logits] = H W + b, W in R^{d x 2}.  Hai classifier (start/end) tren cung H,
    moi classifier cho MOT phan phoi softmax tren cac vi tri T (khong phai per-token binary)."""

    def __init__(self, model_name: str = MODEL_NAME, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)

    def hidden_states(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return self.encoder(**kw).last_hidden_state

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden_states(input_ids, attention_mask, token_type_ids)
        logits = self.qa_outputs(self.dropout(h)).float()                 # [B,T,2]
        start, end = logits[..., 0], logits[..., 1]
        pad = attention_mask == 0                                         # padding khong duoc nhan xac suat
        return start.masked_fill(pad, NEG), end.masked_fill(pad, NEG)


def qa_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_pos: torch.Tensor, end_pos: torch.Tensor) -> torch.Tensor:
    """L = (L_start + L_end) / 2,  L_start = -log P_start(i),  L_end = -log P_end(j)  (cross-entropy tren T vi tri), trung binh theo batch."""
    ce = nn.functional.cross_entropy
    return (ce(start_logits, start_pos) + ce(end_logits, end_pos)) / 2


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def _inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    kw = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
    if "token_type_ids" in batch:
        kw["token_type_ids"] = batch["token_type_ids"].to(device)
    return kw


# --------------------------------------------------------------------------- #
# Inference + postprocess
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_features(model: nn.Module, feats: QAFeatures, pad_id: int, device: torch.device, use_amp: bool = True,
                     batch_size: int = 32, compute_loss: bool = False) -> dict[str, Any]:
    """Logits start/end cho moi feature (list[np.ndarray]) va (tuy chon) loss QA trung binh tren feature co nhan."""
    model.eval()
    order = sorted(range(len(feats)), key=lambda i: len(feats.feats[i]["input_ids"]))
    loader = DataLoader(torch.utils.data.Subset(feats, order), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(pad_id, feats.has_token_type))
    S: list[Any] = [None] * len(feats)
    E: list[Any] = [None] * len(feats)
    loss_sum, n = 0.0, 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            s, e = model(**_inputs(batch, device))
        if compute_loss:
            sp, ep = batch["start_positions"].to(device), batch["end_positions"].to(device)
            loss_sum += float(qa_loss(s, e, sp, ep)) * len(sp)
            n += len(sp)
        s, e = s.cpu().numpy(), e.cpu().numpy()
        for row, idx in enumerate(batch["idx"]):
            L = len(feats.feats[idx]["input_ids"])
            S[idx], E[idx] = s[row, :L], e[row, :L]
    out = {"start": S, "end": E}
    if compute_loss:
        out["loss"] = loss_sum / max(n, 1)
    return out


def postprocess(feats: QAFeatures, contexts: Sequence[str], start_logits: Sequence[np.ndarray], end_logits: Sequence[np.ndarray],
                n_best: int = N_BEST, max_answer_len: int = MAX_ANSWER_LEN) -> list[dict[str, Any]]:
    """Moi example: gom cac window cua no, voi moi window lay n-best start x n-best end (chi token context, end >= start,
    do dai <= max_answer_len token), score = start_logit + end_logit, chon span tot nhat qua TAT CA window.
    Tra ve {'text','start_char','end_char','score','null_score','feature','start_tok','end_tok'}; null_score = min_w (s_CLS + e_CLS)."""
    by_ex: dict[int, list[int]] = collections.defaultdict(list)
    for i, f in enumerate(feats.feats):
        by_ex[f["example_idx"]].append(i)
    out: list[dict[str, Any]] = []
    for ex in range(feats.n_examples):
        best: dict[str, Any] | None = None
        null = float("inf")
        for fi in by_ex[ex]:
            f, s, e = feats.feats[fi], start_logits[fi], end_logits[fi]
            offs = f["offsets"]
            null = min(null, float(s[0] + e[0]))
            s_idx = np.argsort(s)[::-1][:n_best]
            e_idx = np.argsort(e)[::-1][:n_best]
            for a in s_idx:
                if offs[a] is None:
                    continue
                for b in e_idx:
                    if offs[b] is None or b < a or b - a + 1 > max_answer_len:
                        continue
                    sc = float(s[a] + e[b])
                    if best is None or sc > best["score"]:
                        best = {"score": sc, "feature": fi, "start_tok": int(a), "end_tok": int(b),
                                "start_char": offs[a][0], "end_char": offs[b][1]}
        if best is None:                    # khong co span hop le (hiem) -> chuoi rong
            best = {"score": float("-inf"), "feature": by_ex[ex][0], "start_tok": 0, "end_tok": 0, "start_char": 0, "end_char": 0}
        best["text"] = contexts[ex][best["start_char"]:best["end_char"]]
        best["null_score"] = null
        out.append(best)
    return out


def predict(model: nn.Module, tokenizer, questions: Sequence[str], contexts: Sequence[str], max_len: int = 384, stride: int = 128,
            device: torch.device | None = None, use_amp: bool = True, batch_size: int = 64, n_best: int = N_BEST,
            max_answer_len: int = MAX_ANSWER_LEN, return_features: bool = False):
    """Du doan answer span cho cac cap (question, context) tuy y (dung cho experiment + document QA). Co sliding window."""
    device = device or get_device()
    feats = QAFeatures(tokenizer, questions, contexts, max_len, stride)
    fw = forward_features(model, feats, tokenizer.pad_token_id, device, use_amp, batch_size)
    preds = postprocess(feats, contexts, fw["start"], fw["end"], n_best, max_answer_len)
    return (preds, feats, fw) if return_features else preds


def default_stride(max_len: int) -> int:
    """stride (so token overlap giua hai window) = max_len // 3: 128 -> 42, 256 -> 85, 384 -> 128 (dung gia tri de cho 384)."""
    return max_len // 3


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def default_config(**overrides: Any) -> dict[str, Any]:
    cfg = {"model_name": MODEL_NAME, "max_len": 384, "stride": None, "batch_size": 16, "grad_accum": 1, "lr": 3e-5, "epochs": 2,
           "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.1, "fp16": True, "grad_clip": 1.0, "seed": SEED,
           "pretrained": True, "eval_per_epoch": 4, "n_train": N_TRAIN, "n_val": N_VAL, "n_best": N_BEST, "max_answer_len": MAX_ANSWER_LEN}
    cfg.update(overrides)
    if cfg["stride"] is None:
        cfg["stride"] = default_stride(cfg["max_len"])
    return cfg


def run_name(cfg: dict[str, Any]) -> str:
    return (f"{'ft' if cfg['pretrained'] else 'scratch'}_{cfg['model_name'].replace('/', '-')}_len{cfg['max_len']}"
            f"_lr{cfg['lr']:g}_ep{cfg['epochs']}_s{cfg['seed']}")


def eval_split(model, tok, split: Sequence[dict[str, Any]], cfg: dict[str, Any], device, use_amp: bool) -> dict[str, Any]:
    """Val/test: EM, F1 (max tren gold answers) + prediction theo example. Loss tren feature co nhan chi tinh duoc khi co (start,end) char."""
    questions = [r["question"] for r in split]
    contexts = [r["context"] for r in split]
    answers = [(r["answers"]["answer_start"][0], r["answers"]["answer_start"][0] + len(r["answers"]["text"][0])) for r in split]
    feats = QAFeatures(tok, questions, contexts, cfg["max_len"], cfg["stride"], answers)       # co nhan (gold dau tien) de tinh val loss
    fw = forward_features(model, feats, tok.pad_token_id, device, use_amp, compute_loss=True)
    preds = postprocess(feats, contexts, fw["start"], fw["end"], cfg["n_best"], cfg["max_answer_len"])
    golds = [r["answers"]["text"] for r in split]
    m = evaluate_predictions([p["text"] for p in preds], golds)
    m["loss"] = fw["loss"]
    m["n_features"] = len(feats)
    return {"metrics": m, "preds": preds}


def run_experiment(cfg: dict[str, Any], name: str | None = None, force: bool = False, save_model: bool = False,
                   verbose: bool = True) -> dict[str, Any]:
    """Train 1 cau hinh. Checkpoint chon theo val F1 (khong dung test); test (= SQuAD dev) chi danh gia 1 lan cho checkpoint do.
    Cache: runs/<name>/result.json (+ preds_test.json, model.pt neu save_model)."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    name = name or run_name(cfg)
    rdir = RUNS_DIR / name
    if (rdir / "result.json").exists() and not force:
        return json.loads((rdir / "result.json").read_text(encoding="utf8"))
    rdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = bool(cfg["fp16"]) and device.type == "cuda"
    data = make_splits(cfg["n_train"], cfg["n_val"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tr = data["train"]
    train_feats = QAFeatures(tok, [r["question"] for r in tr], [r["context"] for r in tr], cfg["max_len"], cfg["stride"],
                             [(r["answers"]["answer_start"][0], r["answers"]["answer_start"][0] + len(r["answers"]["text"][0])) for r in tr])
    micro = cfg["batch_size"] // cfg["grad_accum"]
    train_loader = make_loader(train_feats, tok.pad_token_id, micro, True, cfg["seed"])
    model = QAModel(cfg["model_name"], pretrained=cfg["pretrained"], dropout=cfg["dropout"]).to(device)
    total_p, train_p = count_parameters(model)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps = (len(train_loader) // cfg["grad_accum"]) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, best_state = [], None, None
    steps_per_epoch = len(train_loader) // cfg["grad_accum"]
    eval_steps = sorted({max(1, round(k * steps_per_epoch / cfg["eval_per_epoch"])) + (e - 1) * steps_per_epoch
                         for e in range(1, cfg["epochs"] + 1) for k in range(1, cfg["eval_per_epoch"] + 1)})
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    train_time_only = 0.0                        # thoi gian TRAIN thuan (khong gom eval validation)
    step_losses: list[float] = []                # train loss tung buoc (optimizer step) - cho learning curve chi tiet
    seg_losses: list[float] = []
    opt_step, t_seg = 0, time.perf_counter()
    stop = False
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                s, e = model(**_inputs(batch, device))
            loss = qa_loss(s, e, batch["start_positions"].to(device), batch["end_positions"].to(device))
            scaler.scale(loss / cfg["grad_accum"]).backward()
            acc += float(loss.detach())
            if step % cfg["grad_accum"] != 0:
                continue
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            step_losses.append(acc / cfg["grad_accum"])
            seg_losses.append(acc / cfg["grad_accum"])
            acc = 0.0
            if opt_step in eval_steps:
                sync()
                train_time_only += time.perf_counter() - t_seg
                val = eval_split(model, tok, data["validation"], cfg, device, use_amp)["metrics"]
                row = {"epoch": opt_step / steps_per_epoch, "step": opt_step, "train_loss": float(np.mean(seg_losses)), "val_loss": val["loss"],
                       "val_em": val["em"], "val_f1": val["f1"], "elapsed_s": time.perf_counter() - t0}
                history.append(row)
                seg_losses = []
                if verbose:
                    print(f"[{name}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
                if best is None or row["val_f1"] > best["val_f1"]:
                    best = row
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                sync()
                t_seg = time.perf_counter()
    train_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    model.load_state_dict(best_state)
    val = eval_split(model, tok, data["validation"], cfg, device, use_amp)
    t_test = time.perf_counter()
    test = eval_split(model, tok, data["test"], cfg, device, use_amp)
    test_time = time.perf_counter() - t_test
    result = {"name": name, "config": cfg, "params_total": total_p, "params_trainable": train_p,
              "train_time_s": train_time, "train_time_only_s": train_time_only, "test_eval_time_s": test_time, "peak_mem_mb": peak_mem,
              "best_epoch": best["epoch"], "best_step": best["step"], "steps_per_epoch": steps_per_epoch, "history": history,
              "train_loss_steps": [round(x, 5) for x in step_losses], "train_feature_stats": train_feats.stats(),
              "val": val["metrics"], "test": test["metrics"], "hardware": hardware_info()}
    slim = [{k: p[k] for k in ("text", "start_char", "end_char", "score", "null_score", "feature")} for p in test["preds"]]
    (rdir / "preds_test.json").write_text(json.dumps({"ids": [r["id"] for r in data["test"]], "preds": slim}), encoding="utf8")
    (rdir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    if save_model:
        torch.save(best_state, rdir / "model.pt")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_trained(name: str, device: torch.device | None = None):
    """Nap model da luu (runs/<name>/model.pt) + tokenizer + config."""
    from transformers import AutoTokenizer
    rdir = RUNS_DIR / name
    cfg = json.loads((rdir / "result.json").read_text(encoding="utf8"))["config"]
    device = device or get_device()
    model = QAModel(cfg["model_name"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(torch.load(rdir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg["model_name"]), cfg


# --------------------------------------------------------------------------- #
# Grid + CLI
# --------------------------------------------------------------------------- #
GRID_SEEDS = (42, 43, 44)


def build_grid() -> dict[str, dict[str, Any]]:
    """Exp 1: max_len 128 / 256 / 384 (seed 42, cac thanh phan khac co dinh); Exp 2: max_len 384 them seed 43, 44 de do nhieu seed."""
    grid = {}
    for L in (384, 256, 128):
        c = default_config(max_len=L, seed=42)
        grid[run_name(c)] = c
    for s in GRID_SEEDS[1:]:
        c = default_config(max_len=384, seed=s)
        grid[run_name(c)] = c
    return grid


MAIN_RUN = run_name(default_config(max_len=384, seed=42))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=N_TRAIN)
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run_grid", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    if a.run_grid:
        for name, cfg in build_grid().items():
            cfg.update(grad_accum=a.grad_accum)
            r = run_experiment(cfg, name, force=a.force, save_model=(name == MAIN_RUN))
            print(name, "test EM/F1 =", round(r["test"]["em"], 4), round(r["test"]["f1"], 4), flush=True)
        return
    cfg = default_config(max_len=a.max_len, stride=a.stride, seed=a.seed, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size,
                         grad_accum=a.grad_accum, n_train=a.n_train, model_name=a.model_name)
    r = run_experiment(cfg, force=a.force, save_model=True)
    print(json.dumps({k: r["test"][k] for k in ("em", "f1")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
