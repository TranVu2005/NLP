"""LAB 08 - Pretrained Transformer (DistilBERT) cho sentiment classification.

Pipeline (muc 13 cua de):
    text -> tokenizer(padding, truncation) -> encoder -> [CLS] (last_hidden_state[:, 0])
         -> Dropout -> Linear -> logits -> softmax

`PretrainedClassifier` tu viet (AutoModel + head tu viet), KHONG dung
AutoModelForSequenceClassification. Tham so:
    freeze_encoder : dong bang encoder (feature extraction)
    pretrained     : False = cung kien truc, khoi tao ngau nhien tu AutoConfig
    pooling        : "cls" (mac dinh) hoac "mean"

CLI vi du:
    python pretrained_classification.py --epochs 3 --lr 2e-5 --max_len 128
    python pretrained_classification.py --freeze --lr 1e-3
    python pretrained_classification.py --scratch --train_frac 0.25
    python pretrained_classification.py --train_domain A --test_domain B

LUU Y Windows: phai import pandas TRUOC torch (neu khong process crash 0xC0000374).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # noqa: I001  (bat buoc import truoc torch tren Windows)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
LAB05 = HERE.parent / "Lab05"
RUNS_DIR = HERE / "runs"
DATA_CACHE = HERE / "data_cache"

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
YELP_ID = "fancyzhx/yelp_polarity"
LABEL_NAMES = ("negative", "positive")


# --------------------------------------------------------------------------- #
# Reproducibility / env
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_vram_mb"] = round(props.total_memory / 2**20)
    return info


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Bo the HTML `<br />` cua IMDb (khong mang nghia) va gop khoang trang."""
    return _WS.sub(" ", _BR.sub(" ", str(text))).strip()


def text_key(text: Any) -> str:
    """Khoa chuan hoa de kiem tra trung lap train<->val<->test."""
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def load_domain_a() -> dict[str, pd.DataFrame]:
    """IMDb: dung NGUYEN split cua LAB 05 (31,879 / 7,904 / 10,000, seed 42)."""
    sys.path.insert(0, str(LAB05))
    import sequence_models as sm  # LAB 05

    raw_train, raw_test, source = sm.load_imdb_dataset(LAB05 / "data")
    train, val, test, report = sm.make_splits(raw_train, raw_test)
    out = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        out[name] = pd.DataFrame(
            {"text": frame["text"].map(clean_text), "label": frame["label"].astype(int),
             "sample_id": frame["sample_id"]}
        )
    out["_meta"] = {"source": source, "lab05_leakage_report": report}  # type: ignore[assignment]
    return out


def load_domain_b(n_train: int = 31_879, n_val: int = 7_904, n_test: int = 10_000, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Yelp polarity (review nha hang/dich vu): subset co dinh, can bang nhan, khop kich thuoc voi domain A.

    train/val lay tu split 'train' cua Yelp, test lay tu split 'test'; loai trung lap giua cac tap.
    """
    from datasets import load_dataset

    ds = load_dataset(YELP_ID)
    rng = np.random.RandomState(seed)

    def balanced_pick(df: pd.DataFrame, n: int, exclude: set[str]) -> pd.DataFrame:
        picks = []
        for label, count in ((0, n // 2), (1, n - n // 2)):
            sub = df[df["label"] == label]
            order = rng.permutation(len(sub))
            chosen, seen = [], set()
            for pos in order:
                row = sub.iloc[pos]
                key = text_key(row["text"])
                if key in exclude or key in seen:
                    continue
                seen.add(key)
                chosen.append(pos)
                if len(chosen) == count:
                    break
            picks.append(sub.iloc[chosen])
        return pd.concat(picks).reset_index(drop=True)

    train_all = ds["train"].to_pandas().reset_index().rename(columns={"index": "orig_idx"})
    test_all = ds["test"].to_pandas().reset_index().rename(columns={"index": "orig_idx"})
    # thu hep pool de nhanh: 200k dong ngau nhien cua train (seed co dinh)
    pool_idx = rng.permutation(len(train_all))[:200_000]
    train_pool = train_all.iloc[pool_idx].reset_index(drop=True)

    test_df = balanced_pick(test_all, n_test, set())
    test_keys = {text_key(t) for t in test_df["text"]}
    train_df = balanced_pick(train_pool, n_train, test_keys)
    train_keys = {text_key(t) for t in train_df["text"]}
    val_df = balanced_pick(train_pool, n_val, test_keys | train_keys)

    out = {}
    for name, frame, prefix in (("train", train_df, "yelp_train"), ("val", val_df, "yelp_val"), ("test", test_df, "yelp_test")):
        out[name] = pd.DataFrame(
            {"text": frame["text"].map(clean_text), "label": frame["label"].astype(int),
             "sample_id": [f"{prefix}:{int(i)}" for i in frame["orig_idx"]]}
        )
    out["_meta"] = {"source": f"Hugging Face {YELP_ID}", "sizes": {k: len(out[k]) for k in ("train", "val", "test")}}  # type: ignore[assignment]
    return out


_DATA: dict[str, dict[str, Any]] = {}


def get_data(domain: str) -> dict[str, Any]:
    """Cache trong process. domain in {'A','B'}."""
    if domain not in _DATA:
        _DATA[domain] = load_domain_a() if domain == "A" else load_domain_b()
    return _DATA[domain]


def check_leakage(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """So van ban chuan hoa trung nhau giua cac cap tap. Phai = 0."""
    keys = {name: {text_key(t) for t in df["text"]} for name, df in frames.items()}
    names = list(keys)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}&{b}"] = len(keys[a] & keys[b])
    return out


def stratified_subset(labels: Sequence[int], frac: float, seed: int) -> list[int]:
    """Subset co dinh theo (frac, seed), can bang nhan."""
    if frac >= 1.0:
        return list(range(len(labels)))
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    chosen: list[int] = []
    for label in (0, 1):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        chosen.extend(idx[: max(1, math.ceil(len(idx) * frac))].tolist())
    return sorted(chosen)


class EncodedDataset(Dataset):
    """Tokenize 1 lan (truncation); padding dong theo batch trong collate_fn."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int) -> None:
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)
        self.ids: list[list[int]] = enc["input_ids"]
        self.labels = [int(x) for x in labels]
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        return self.ids[i], self.labels[i]

    @property
    def truncated_fraction(self) -> float:
        return float(np.mean([len(x) >= self.max_len for x in self.ids]))

    def collate(self, batch):
        longest = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), longest), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": mask,
                "labels": torch.tensor([label for _, label in batch], dtype=torch.long)}


def make_loader(ds: EncodedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ds.collate, generator=gen, num_workers=0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PretrainedClassifier(nn.Module):
    """AutoModel (encoder) + head tu viet: pooling -> Dropout -> Linear."""

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = 2, dropout: float = 0.1,
                 freeze_encoder: bool = False, pretrained: bool = True, pooling: str = "cls") -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        config = AutoConfig.from_pretrained(model_name)
        # pretrained=False: cung kien truc/cau hinh, trong so ngau nhien (kiem soat bien)
        self.encoder = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(config)
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):  # frozen encoder luon o eval (tat dropout) = feature extractor thuan
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(self.pool(hidden, attention_mask)))  # logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict[str, Any]:
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        ids, mask, y = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(ids, mask)
        logits = logits.float()
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return {"loss": loss_sum / n, "probs": probs, "labels": labels, **metrics_from_probs(probs, labels)}


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, pred)), "precision": float(p), "recall": float(r),
            "f1": float(f), "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def default_config(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_name": MODEL_NAME, "pooling": "cls", "pretrained": True, "freeze": False,
        "max_len": 128, "batch_size": 32, "grad_accum": 1, "lr": 2e-5, "epochs": 3,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.1, "max_grad_norm": 1.0,
        "train_frac": 1.0, "train_domain": "A", "test_domains": ["A"], "seed": SEED, "fp16": True,
    }
    cfg.update(overrides)
    return cfg


DATASET_NAMES = {"A": "IMDb (LAB 05 split, seed 42, stratified val 20% + duplicate-leak cleaning)",
                 "B": f"Yelp polarity ({YELP_ID}), fixed balanced subset, seed 42"}


def make_run_config(result: dict[str, Any]) -> dict[str, Any]:
    """Bang tai lap (muc 23) tu result.json: Model, Tokenizer, Dataset, splits, max_len, batch, lr, epochs, optimizer, seed, hardware, time."""
    cfg = result["cfg"]
    tests = cfg["test_domains"]
    return {
        "Model": cfg["model_name"] + ("" if cfg["pretrained"] else " (random init, same architecture via AutoConfig)")
                 + (" [encoder frozen]" if cfg["freeze"] else ""),
        "Tokenizer": cfg["model_name"] + " (WordPiece, same checkpoint)",
        "Dataset": f"train: {DATASET_NAMES[cfg['train_domain']]}",
        "Train split": f"{result['n_train']} (train_frac={cfg['train_frac']} of domain {cfg['train_domain']} train)",
        "Validation split": f"{result['n_val']} (domain {cfg['train_domain']} val, fixed)",
        "Test split": {d: f"{result['test'][d]['n']} ({DATASET_NAMES[d]})" for d in tests},
        "Max sequence length": cfg["max_len"], "Batch size": cfg["batch_size"] * cfg["grad_accum"],
        "Learning rate": cfg["lr"], "Epochs": cfg["epochs"],
        "Optimizer": f"AdamW(weight_decay={cfg['weight_decay']}) + linear warmup({cfg['warmup_ratio']})/decay, grad_clip={cfg['max_grad_norm']}, fp16={cfg['fp16']}",
        "Random seed": cfg["seed"], "Hardware": hardware_info(), "Training time (s)": round(result["train_time_s"], 1),
        "Peak memory (MB)": round(result["peak_mem_mb"], 1), "best_epoch (by val loss)": result["best_epoch"],
        "cfg": cfg, "params_total": result["params_total"], "params_trainable": result["params_trainable"],
    }


def run_experiment(cfg: dict[str, Any], name: str, force: bool = False, keep_model: bool = False,
                   verbose: bool = True, save_model: bool = False) -> dict[str, Any]:
    """Train + eval 1 cau hinh. Ket qua cache o runs/<name>/ (result.json, run_config.json, probs_*.npy)."""
    out_dir = RUNS_DIR / name
    result_path = out_dir / "result.json"
    if result_path.exists() and not force and not keep_model and (not save_model or (out_dir / "model.pt").exists()):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("cfg") == cfg:
            if verbose:
                print(f"[cache] {name}: f1={cached['test'][cfg['test_domains'][0]]['f1']:.4f}")
            return cached

    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    use_amp = bool(cfg["fp16"]) and device.type == "cuda"
    seed = int(cfg["seed"])
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    data = get_data(cfg["train_domain"])
    train_df, val_df = data["train"], data["val"]
    idx = stratified_subset(train_df["label"].tolist(), cfg["train_frac"], seed)
    train_sub = train_df.iloc[idx]

    train_ds = EncodedDataset(train_sub["text"], train_sub["label"], tokenizer, cfg["max_len"])
    val_ds = EncodedDataset(val_df["text"], val_df["label"], tokenizer, cfg["max_len"])
    test_ds = {d: EncodedDataset(get_data(d)["test"]["text"], get_data(d)["test"]["label"], tokenizer, cfg["max_len"])
               for d in cfg["test_domains"]}

    model = PretrainedClassifier(cfg["model_name"], dropout=cfg["dropout"], freeze_encoder=cfg["freeze"],
                                 pretrained=cfg["pretrained"], pooling=cfg["pooling"]).to(device)
    params_total, params_trainable = count_parameters(model)

    train_loader = make_loader(train_ds, cfg["batch_size"], True, seed)
    val_loader = make_loader(val_ds, 128, False, seed)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [
        {"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_state, best_val, best_epoch = None, float("inf"), -1
    train_time = 0.0
    wall0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        loss_sum, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            ids, mask, y = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), y)
            scaler.scale(loss / cfg["grad_accum"]).backward()
            if step % cfg["grad_accum"] == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for _, p in named], cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            n_batches += 1
        _sync(device)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time
        val = evaluate(model, val_loader, device, use_amp)
        history.append({"epoch": epoch, "train_loss": loss_sum / n_batches, "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"], "val_f1": val["f1"], "epoch_time_s": epoch_time})
        if verbose:
            print(f"[{name}] ep{epoch} train_loss={loss_sum / n_batches:.4f} val_loss={val['loss']:.4f} "
                  f"val_acc={val['accuracy']:.4f} ({epoch_time:.0f}s)", flush=True)
        if val["loss"] < best_val:  # chon checkpoint theo val loss (khong dung test)
            best_val, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if (not cfg["freeze"]) or not k.startswith("encoder.")}
    peak_train_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else _rss_mb()
    wall = time.perf_counter() - wall0

    model.load_state_dict(best_state, strict=False)
    if save_model:  # fp16 de nho; da .gitignore
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_dir / "model.pt")
    test_metrics: dict[str, Any] = {}
    for d, ds in test_ds.items():
        res = evaluate(model, make_loader(ds, 128, False, seed), device, use_amp)
        np.save(out_dir / f"probs_test_{d}.npy", res.pop("probs").astype(np.float32))
        res.pop("labels")
        res["n"] = len(ds)
        res["truncated_fraction"] = ds.truncated_fraction
        test_metrics[d] = res
    val_best = evaluate(model, val_loader, device, use_amp)

    result = {
        "name": name, "cfg": cfg, "params_total": params_total, "params_trainable": params_trainable,
        "train_time_s": train_time, "total_wall_s": wall, "peak_mem_mb": peak_train_mb,
        "n_train": len(train_ds), "n_val": len(val_ds), "train_truncated_fraction": train_ds.truncated_fraction,
        "best_epoch": best_epoch, "history": history,
        "val": {k: v for k, v in val_best.items() if k not in ("probs", "labels")}, "test": test_metrics,
    }
    run_config = make_run_config(result)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if keep_model:
        result["_model"] = model
        result["_tokenizer"] = tokenizer
    else:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def load_model(name: str) -> tuple["PretrainedClassifier", Any, dict[str, Any]]:
    """Nap model da luu (runs/<name>/model.pt) + tokenizer + result.json."""
    result = json.loads((RUNS_DIR / name / "result.json").read_text(encoding="utf-8"))
    cfg = result["cfg"]
    model = PretrainedClassifier(cfg["model_name"], dropout=cfg["dropout"], pretrained=False, pooling=cfg["pooling"])
    state = torch.load(RUNS_DIR / name / "model.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(get_device()).eval(), AutoTokenizer.from_pretrained(cfg["model_name"]), result


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


@torch.no_grad()
def predict_texts(model: PretrainedClassifier, tokenizer, texts: Sequence[str], max_len: int = 128,
                  batch_size: int = 64) -> np.ndarray:
    """text -> tokenizer(padding, truncation, return_tensors='pt') -> softmax probs (N, 2)."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        out.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"]).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Luoi thi nghiem (dung chung cho notebook va CLI --run_grid)
# --------------------------------------------------------------------------- #
FRACS = (0.1, 0.25, 0.5, 1.0)
MAX_LENS = (64, 128, 256)
EXTRA_SEEDS = (43, 44)
SCRATCH_LR_SWEEP = (5e-5, 1e-4, 3e-4)
FT_LR, FROZEN_LR = 2e-5, 1e-3
SAVE_MODEL_RUNS = ("ft_A_len64_frac1.0_s42", "ft_A_len128_frac1.0_s42", "ft_A_len256_frac1.0_s42")
SCRATCH_LR = 5e-5  # chon bang val loss o lrsweep_scratch (5e-5: val loss 0.448 < 1e-4: 0.475 < 3e-4: 0.534); ghi lai trong README


def run_name(kind: str, frac: float = 1.0, max_len: int = 128, seed: int = SEED, domain: str = "A") -> str:
    return f"{kind}_{domain}_len{max_len}_frac{frac}_s{seed}"


def build_grid(scratch_lr: float = SCRATCH_LR) -> dict[str, dict[str, Any]]:
    """name -> cfg cho moi run. Run chinh (ft/scratch @128, 100%) dung chung cho Exp 1,2,3,4,5."""
    g: dict[str, dict[str, Any]] = {}
    both = ["A", "B"]
    # lr sweep cho scratch (chon theo val loss, khong theo test) tren 25% data
    for lr in SCRATCH_LR_SWEEP:
        g[f"lrsweep_scratch_lr{lr:g}"] = default_config(pretrained=False, lr=lr, train_frac=0.25)
    # main + Exp 1/2/4/5
    g[run_name("ft")] = default_config(lr=FT_LR, test_domains=both)
    g[run_name("scratch")] = default_config(pretrained=False, lr=scratch_lr, test_domains=both)
    g[run_name("frozen")] = default_config(freeze=True, lr=FROZEN_LR)
    for t in MAX_LENS:
        if t != 128:
            g[run_name("ft", max_len=t)] = default_config(lr=FT_LR, max_len=t)
    g[run_name("ft", domain="B")] = default_config(lr=FT_LR, train_domain="B", test_domains=["B", "A"])
    # Exp 3 data efficiency (seed 42, roi seed 43/44)
    for seed in (SEED,) + EXTRA_SEEDS:
        for frac in FRACS:
            if seed == SEED and frac == 1.0:
                continue  # = run chinh
            g[run_name("ft", frac=frac, seed=seed)] = default_config(lr=FT_LR, train_frac=frac, seed=seed,
                                                                      test_domains=both if frac == 1.0 else ["A"])
            g[run_name("scratch", frac=frac, seed=seed)] = default_config(
                pretrained=False, lr=scratch_lr, train_frac=frac, seed=seed,
                test_domains=both if frac == 1.0 else ["A"])
    return g


def write_manifest() -> None:
    """Luu sample_id cua cac split de tai lap/kiem tra (data_cache/splits_manifest.json)."""
    DATA_CACHE.mkdir(exist_ok=True)
    manifest = {}
    for d in ("A", "B"):
        data = get_data(d)
        manifest[d] = {k: data[k]["sample_id"].tolist() for k in ("train", "val", "test")}
        manifest[d]["meta"] = data["_meta"]
    (DATA_CACHE / "splits_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LAB 08 pretrained classifier")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="mac dinh 2e-5 (fine-tune) / 1e-3 (--freeze)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--freeze", action="store_true", help="dong bang encoder")
    p.add_argument("--scratch", action="store_true", help="khoi tao ngau nhien cung kien truc")
    p.add_argument("--train_domain", choices=["A", "B"], default="A")
    p.add_argument("--test_domain", choices=["A", "B", "both"], default="A")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--name", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--run_grid", action="store_true", help="chay (va cache) toan bo luoi thi nghiem cua LAB 08")
    p.add_argument("--only", default=None, help="voi --run_grid: chi chay ten chua chuoi nay")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    if a.run_grid:
        for name, cfg in build_grid().items():
            if a.only is None or a.only in name:
                run_experiment(cfg, name, force=a.force, save_model=name in SAVE_MODEL_RUNS)
        return
    lr = a.lr if a.lr is not None else (1e-3 if a.freeze else 2e-5)
    tests = ["A", "B"] if a.test_domain == "both" else [a.test_domain]
    cfg = default_config(model_name=a.model_name, pooling=a.pooling, pretrained=not a.scratch, freeze=a.freeze,
                         max_len=a.max_len, batch_size=a.batch_size, grad_accum=a.grad_accum, lr=lr,
                         epochs=a.epochs, train_frac=a.train_frac, train_domain=a.train_domain,
                         test_domains=tests, seed=a.seed)
    name = a.name or ("cli_" + ("scratch" if a.scratch else "frozen" if a.freeze else "ft")
                      + f"_{a.train_domain}_len{a.max_len}_frac{a.train_frac}_s{a.seed}")
    res = run_experiment(cfg, name, force=a.force)
    print(json.dumps({"name": name, "params_total": res["params_total"], "params_trainable": res["params_trainable"],
                      "train_time_s": round(res["train_time_s"], 1), "peak_mem_mb": round(res["peak_mem_mb"], 1),
                      "test": {d: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
                               for d, m in res["test"].items()}}, indent=2))
    print(f"run_config.json -> {RUNS_DIR / name / 'run_config.json'}")


if __name__ == "__main__":
    main()
