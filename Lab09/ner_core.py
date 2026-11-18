"""LAB 09 - Loi dung chung cho NER tren CoNLL-2003: du lieu, BIO, label alignment,
metric entity-level, TokenClassifier (AutoModel + head tu viet) va vong train/eval.

Pipeline (muc 14 cua de):
    words -> tokenizer(is_split_into_words=True) -> input_ids/attention_mask/word_ids
          -> alignment -> token_labels (-100 = IGNORE)
          -> pretrained encoder -> H [B,T,d] -> Dropout -> Linear -> logits Z [B,T,C]
          -> CrossEntropy(ignore_index=-100) -> argmax -> nhan BIO

`TokenClassifier` tu viet (AutoModel + head tu viet), KHONG dung
AutoModelForTokenClassification va KHONG dung pipeline("ner").
Metric entity-level tu cai dat theo quy uoc conlleval (khop tuyet doi ranh gioi + loai).

CLI:
    python ner_core.py --strategy A --seed 42            # 1 run (cache vao runs/)
    python ner_core.py --run_grid                        # toan bo run cua bao cao

LUU Y Windows: phai import pandas TRUOC torch (neu khong process co the crash).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # noqa: F401,I001  (bat buoc import truoc torch tren Windows)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
DATA_CACHE = HERE / "data_cache"

SEED = 42
MODEL_NAME = "bert-base-cased"
DATASET_ID = "eriktks/conll2003"
DATASET_REVISION = "refs/convert/parquet"      # ban parquet cua CoNLL-2003 (datasets>=4 khong chay script)
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
IGNORE_INDEX = -100


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
# Data: CoNLL-2003
# --------------------------------------------------------------------------- #
def load_conll(use_cache: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Tra ve {'train','validation','test'}: moi phan tu {'id','tokens','tags'} (tags la chuoi nhan BIO).
    Lan dau tai tu HF Hub roi cache thanh jsonl trong data_cache/ (tai lap offline)."""
    DATA_CACHE.mkdir(exist_ok=True)
    out: dict[str, list[dict[str, Any]]] = {}
    files = {s: DATA_CACHE / f"conll2003_{s}.jsonl" for s in ("train", "validation", "test")}
    if use_cache and all(f.exists() for f in files.values()):
        for s, f in files.items():
            out[s] = [json.loads(line) for line in f.read_text(encoding="utf8").splitlines()]
        return out
    import datasets
    ds = datasets.load_dataset(DATASET_ID, revision=DATASET_REVISION)
    names = ds["train"].features["ner_tags"].feature.names
    assert list(names) == LABELS, names
    for s in files:
        out[s] = [{"id": r["id"], "tokens": list(r["tokens"]), "tags": [names[t] for t in r["ner_tags"]]} for r in ds[s]]
        files[s].write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out[s]), encoding="utf8")
    return out


# --------------------------------------------------------------------------- #
# BIO + entity-level metrics (conlleval semantics)
# --------------------------------------------------------------------------- #
def tags_to_entities(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """Chuoi nhan BIO -> [(type, start, end_exclusive)]. Quy uoc conlleval: `I-X` khong sau `B-X`/`I-X` van bat dau
    mot entity moi (khong bo qua), `B-X` luon bat dau entity moi, doi loai trong I- tach entity."""
    ents: list[tuple[str, int, int]] = []
    start: int | None = None
    typ: str | None = None
    for i, t in enumerate(list(tags) + ["O"]):
        prefix, ttype = ("O", None) if t == "O" else t.split("-", 1)
        if start is not None and (prefix in ("O", "B") or ttype != typ):
            ents.append((typ, start, i))  # type: ignore[arg-type]
            start, typ = None, None
        if prefix == "B" or (prefix == "I" and start is None):
            start, typ = i, ttype
    return ents


def entities_to_tags(n: int, ents: Sequence[tuple[str, int, int]]) -> list[str]:
    tags = ["O"] * n
    for typ, s, e in ents:
        tags[s] = f"B-{typ}"
        for j in range(s + 1, e):
            tags[j] = f"I-{typ}"
    return tags


def invalid_bio_transitions(tags: Sequence[str]) -> int:
    """So vi tri `I-X` ma truoc do khong phai `B-X`/`I-X` (chuoi BIO khong hop le)."""
    bad, prev = 0, "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            bad += 1
        prev = t
    return bad


def entity_set(tag_seqs: Sequence[Sequence[str]]) -> set[tuple[int, str, int, int]]:
    return {(i, t, s, e) for i, tags in enumerate(tag_seqs) for t, s, e in tags_to_entities(tags)}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def entity_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict[str, dict[str, float]]:
    """Entity-level P/R/F1 (micro) + tung loai + macro. TP = entity khop CA ranh gioi LAN loai."""
    g, p = entity_set(gold), entity_set(pred)
    res: dict[str, dict[str, float]] = {}
    for typ in ENTITY_TYPES:
        gt, pt = {x for x in g if x[1] == typ}, {x for x in p if x[1] == typ}
        res[typ] = _prf(len(gt & pt), len(pt - gt), len(gt - pt))
    res["micro"] = _prf(len(g & p), len(p - g), len(g - p))
    res["macro"] = {k: float(np.mean([res[t][k] for t in ENTITY_TYPES])) for k in ("precision", "recall", "f1")}
    return res


def entity_metrics_by_length(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]],
                             buckets: Sequence[tuple[str, int, int]] = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 10**9))
                             ) -> dict[str, dict[str, float]]:
    """P/R/F1 theo do dai entity (so token). Entity duoc xep nhom theo CHINH do dai cua no
    (gold theo do dai gold, prediction theo do dai prediction; TP thi hai do dai bang nhau)."""
    g, p = entity_set(gold), entity_set(pred)
    res = {}
    for name, lo, hi in buckets:
        gb = {x for x in g if lo <= x[3] - x[2] <= hi}
        pb = {x for x in p if lo <= x[3] - x[2] <= hi}
        res[name] = _prf(len(gb & pb), len(pb - gb), len(gb - pb))
    return res


def categorize_errors(gold: Sequence[str], pred: Sequence[str]) -> list[dict[str, Any]]:
    """Phan loai loi entity cua MOT cau (moi gold entity khong khop + moi pred entity khong chong lap gold):
      missing     : gold entity khong co pred nao chong lap (FN thuan)          -> missing entity
      wrong_type  : cung span, khac loai                                         -> wrong entity type
      boundary    : chong lap, cung loai, khac span                              -> wrong boundary
      boundary+type: chong lap, khac span VA khac loai
      spurious    : pred entity khong chong lap gold nao (FP thuan)              -> spurious entity
    Moi muc: {'category','gold': (type,s,e)|None,'pred': (type,s,e)|None}."""
    g, p = tags_to_entities(gold), tags_to_entities(pred)
    gs, ps = set(g), set(p)
    overlap = lambda a, b: a[1] < b[2] and b[1] < a[2]
    out: list[dict[str, Any]] = []
    for ge in g:
        if ge in ps:
            continue
        ov = [pe for pe in p if overlap(ge, pe)]
        if not ov:
            out.append({"category": "missing", "gold": ge, "pred": None})
            continue
        same_span = [pe for pe in ov if pe[1] == ge[1] and pe[2] == ge[2]]
        same_type = [pe for pe in ov if pe[0] == ge[0]]
        if same_span:
            out.append({"category": "wrong_type", "gold": ge, "pred": same_span[0]})
        elif same_type:
            out.append({"category": "boundary", "gold": ge, "pred": same_type[0]})
        else:
            out.append({"category": "boundary+type", "gold": ge, "pred": ov[0]})
    for pe in p:
        if pe not in gs and not any(overlap(pe, ge) for ge in g):
            out.append({"category": "spurious", "gold": None, "pred": pe})
    return out


def token_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict[str, float]:
    """Token-level: accuracy (gom ca `O`) va P/R/F1 micro tren token KHONG phai `O` (nhan BIO phai khop chinh xac)."""
    n = correct = tp = fp = fn = 0
    for gs, ps in zip(gold, pred):
        for a, b in zip(gs, ps):
            n += 1
            correct += a == b
            if a != "O" and a == b:
                tp += 1
            elif b != "O" and a != b:
                fp += 1
            if a != "O" and a != b:
                fn += 1
    out = _prf(tp, fp, fn)
    out["accuracy"] = correct / n
    out["n_tokens"] = n
    out["majority_O_accuracy"] = sum(a == "O" for gs in gold for a in gs) / n
    return out


def confusion_counts(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> np.ndarray:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for gs, ps in zip(gold, pred):
        for a, b in zip(gs, ps):
            cm[LABEL2ID[a], LABEL2ID[b]] += 1
    return cm


# --------------------------------------------------------------------------- #
# Tokenizer + label alignment
# --------------------------------------------------------------------------- #
def align_labels(word_ids: Sequence[int | None], word_labels: Sequence[str], strategy: str = "A") -> list[int]:
    """word_ids (tu tokenizer) + nhan theo tu -> nhan theo subword token (id; -100 = IGNORE).

    Strategy A: chi subword DAU cua moi tu nhan nhan cua tu; subword tiep theo = -100.
    Strategy B: subword dau giu nhan goc; subword tiep theo nhan `I-X` cung loai neu tu la entity (B-X/I-X), `O` neu la O.
    Special token ([CLS], [SEP], padding; word_id = None) luon la -100.
    """
    assert strategy in ("A", "B"), strategy
    out: list[int] = []
    prev: int | None = None
    for wid in word_ids:
        if wid is None:
            out.append(IGNORE_INDEX)
        elif wid != prev:
            out.append(LABEL2ID[word_labels[wid]])
        elif strategy == "A":
            out.append(IGNORE_INDEX)
        else:
            lab = word_labels[wid]
            out.append(LABEL2ID["I-" + lab[2:]] if lab != "O" else LABEL2ID["O"])
        prev = wid
    return out


def first_subword_positions(word_ids: Sequence[int | None]) -> dict[int, int]:
    """word index -> vi tri subword dau tien (nhung tu bi truncation khong co mat trong dict)."""
    pos: dict[int, int] = {}
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in pos:
            pos[wid] = i
    return pos


class NERDataset(Dataset):
    """Tokenize san toan bo tap (is_split_into_words=True), giu word_ids de decode ve muc tu."""

    def __init__(self, records: Sequence[dict[str, Any]], tokenizer, strategy: str = "A", max_len: int = 256):
        self.records = list(records)
        self.items: list[dict[str, Any]] = []
        enc = tokenizer([r["tokens"] for r in self.records], is_split_into_words=True, truncation=True, max_length=max_len)
        self.n_truncated_sentences = 0
        self.n_truncated_words = 0
        for k, r in enumerate(self.records):
            wids = enc.word_ids(k)
            seen = {w for w in wids if w is not None}
            if len(seen) < len(r["tokens"]):
                self.n_truncated_sentences += 1
                self.n_truncated_words += len(r["tokens"]) - len(seen)
            self.items.append({"input_ids": enc["input_ids"][k], "word_ids": wids,
                               "labels": align_labels(wids, r["tags"], strategy)})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {**self.items[i], "idx": i}


def make_collate(pad_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        T = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        lab = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
            lab[i, :n] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": lab, "idx": [b["idx"] for b in batch]}
    return collate


def make_loader(ds: NERDataset, pad_id: int, batch_size: int, shuffle: bool, seed: int = SEED) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, collate_fn=make_collate(pad_id))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TokenClassifier(nn.Module):
    """H = encoder(x) in R^{B x T x d};  Z = Dropout(H) W + b in R^{B x T x C}. Moi token mot vector logits."""

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = len(LABELS), dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(h))

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy trung binh tren cac token co nhan (label != -100): L = -sum_{i: y_i != IGNORE} log P(y_i|x) / #(i)."""
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Predict / evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_dataset(model: nn.Module, ds: NERDataset, pad_id: int, device: torch.device, use_amp: bool = True,
                    batch_size: int = 64, return_probs: bool = False) -> dict[str, Any]:
    """Du doan muc TU: nhan cua tu = argmax logits tai subword DAU TIEN cua tu (cho ca Strategy A va B).
    Tu bi truncation (khong co subword) duoc dat `O`. Tra ve pred_tags, confidence (max softmax), loss tren token co nhan."""
    model.eval()
    order = sorted(range(len(ds)), key=lambda i: len(ds.items[i]["input_ids"]))
    loader = DataLoader(torch.utils.data.Subset(ds, order), batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id))
    pred_tags: list[list[str] | None] = [None] * len(ds)
    conf: list[list[float] | None] = [None] * len(ds)
    probs_out: list[np.ndarray | None] = [None] * len(ds)
    loss_sum, n_lab = 0.0, 0
    for batch in loader:
        ids, mask, lab = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model(ids, mask)
        logits = logits.float()
        loss_sum += float(nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1),
                                                      ignore_index=IGNORE_INDEX, reduction="sum"))
        n_lab += int((lab != IGNORE_INDEX).sum())
        pr = torch.softmax(logits, -1).cpu().numpy()
        for row, idx in enumerate(batch["idx"]):
            item = ds.items[idx]
            n_words = len(ds.records[idx]["tokens"])
            pos = first_subword_positions(item["word_ids"])
            tags, cf, pw = [], [], np.zeros((n_words, len(LABELS)), dtype=np.float32)
            pw[:, LABEL2ID["O"]] = 1.0
            for w in range(n_words):
                if w in pos:
                    p = pr[row, pos[w]]
                    tags.append(LABELS[int(p.argmax())])
                    cf.append(float(p.max()))
                    pw[w] = p
                else:
                    tags.append("O")
                    cf.append(float("nan"))
            pred_tags[idx], conf[idx], probs_out[idx] = tags, cf, pw
    out: dict[str, Any] = {"pred_tags": pred_tags, "confidence": conf, "loss": loss_sum / max(n_lab, 1)}
    if return_probs:
        out["probs"] = probs_out
    return out


def evaluate_split(model: nn.Module, ds: NERDataset, pad_id: int, device: torch.device, use_amp: bool = True) -> dict[str, Any]:
    pred = predict_dataset(model, ds, pad_id, device, use_amp)
    gold = [r["tags"] for r in ds.records]
    em = entity_metrics(gold, pred["pred_tags"])
    tm = token_metrics(gold, pred["pred_tags"])
    return {"entity": em, "token": tm, "loss": pred["loss"], "pred_tags": pred["pred_tags"], "confidence": pred["confidence"]}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def default_config(**overrides: Any) -> dict[str, Any]:
    cfg = {"model_name": MODEL_NAME, "strategy": "A", "max_len": 256, "batch_size": 16, "lr": 5e-5, "epochs": 3,
           "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.1, "fp16": True, "grad_clip": 1.0, "seed": SEED,
           "pretrained": True}
    cfg.update(overrides)
    return cfg


def run_name(cfg: dict[str, Any]) -> str:
    base = f"{'ft' if cfg['pretrained'] else 'scratch'}_{cfg['model_name'].replace('/', '-')}_{cfg['strategy']}_lr{cfg['lr']:g}_ep{cfg['epochs']}_s{cfg['seed']}"
    return base


def run_experiment(cfg: dict[str, Any], name: str | None = None, force: bool = False, save_model: bool = False,
                   verbose: bool = True) -> dict[str, Any]:
    """Train 1 cau hinh. Checkpoint chon theo val entity-F1 (khong dung test); test chi danh gia 1 lan cho checkpoint do.
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
    data = load_conll()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    dsets = {s: NERDataset(data[s], tok, cfg["strategy"], cfg["max_len"]) for s in data}
    train_loader = make_loader(dsets["train"], tok.pad_token_id, cfg["batch_size"], True, cfg["seed"])
    model = TokenClassifier(cfg["model_name"], dropout=cfg["dropout"], pretrained=cfg["pretrained"]).to(device)
    total_p, train_p = count_parameters(model)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps = len(train_loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, best_state = [], None, None
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        run_loss, n_b = 0.0, 0
        for batch in train_loader:
            ids, mask, lab = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask)
            loss = token_loss(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss)
            n_b += 1
        val = evaluate_split(model, dsets["validation"], tok.pad_token_id, device, use_amp)
        row = {"epoch": epoch, "train_loss": run_loss / n_b, "val_loss": val["loss"], "val_entity_f1": val["entity"]["micro"]["f1"],
               "val_token_acc": val["token"]["accuracy"], "elapsed_s": time.perf_counter() - t0}
        history.append(row)
        if verbose:
            print(f"[{name}] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if best is None or row["val_entity_f1"] > best["val_entity_f1"]:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    model.load_state_dict(best_state)
    val = evaluate_split(model, dsets["validation"], tok.pad_token_id, device, use_amp)
    test = evaluate_split(model, dsets["test"], tok.pad_token_id, device, use_amp)
    result = {"name": name, "config": cfg, "params_total": total_p, "params_trainable": train_p, "train_time_s": train_time,
              "peak_mem_mb": peak_mem, "best_epoch": best["epoch"], "history": history,
              "val": {"entity": val["entity"], "token": val["token"], "loss": val["loss"]},
              "test": {"entity": test["entity"], "token": test["token"], "loss": test["loss"]},
              "truncated": {s: {"sentences": dsets[s].n_truncated_sentences, "words": dsets[s].n_truncated_words} for s in dsets},
              "hardware": hardware_info()}
    (rdir / "preds_test.json").write_text(json.dumps({"pred_tags": test["pred_tags"], "confidence": test["confidence"]}), encoding="utf8")
    (rdir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    if save_model:
        torch.save({k: v for k, v in best_state.items()}, rdir / "model.pt")
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
    model = TokenClassifier(cfg["model_name"], dropout=cfg["dropout"], pretrained=False)
    model.load_state_dict(torch.load(rdir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg["model_name"]), cfg


# --------------------------------------------------------------------------- #
# Grid + CLI
# --------------------------------------------------------------------------- #
GRID_SEEDS = (42, 43, 44)


def build_grid() -> dict[str, dict[str, Any]]:
    """Run chinh: Strategy A x 3 seed (seed 42 luu model.pt cho IE) + Strategy B (seed 42) de phan tich anh huong alignment."""
    grid = {}
    for s in GRID_SEEDS:
        c = default_config(strategy="A", seed=s)
        grid[run_name(c)] = c
    c = default_config(strategy="B", seed=42)
    grid[run_name(c)] = c
    return grid


MAIN_RUN = run_name(default_config(strategy="A", seed=42))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["A", "B"], default="A")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run_grid", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    if a.run_grid:
        for name, cfg in build_grid().items():
            r = run_experiment(cfg, name, force=a.force, save_model=(name == MAIN_RUN))
            print(name, "test entity F1 =", round(r["test"]["entity"]["micro"]["f1"], 4))
        return
    cfg = default_config(strategy=a.strategy, seed=a.seed, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, model_name=a.model_name)
    r = run_experiment(cfg, force=a.force, save_model=True)
    print(json.dumps({k: r["test"]["entity"]["micro"][k] for k in ("precision", "recall", "f1")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
