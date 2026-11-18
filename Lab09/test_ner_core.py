"""Kiem tra nhanh cho ner_core.py (chay: python -m pytest test_ner_core.py -q)."""
import pandas as pd  # noqa: F401  (Windows: pandas truoc torch)
import numpy as np
import pytest
import torch

import ner_core as nc


# ---------------- BIO / entity-level metrics (gia tri tinh tay) ---------------- #
def test_tags_to_entities_basic_and_conlleval_semantics():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "O", "I-ORG", "I-ORG", "B-ORG", "I-LOC"]
    assert nc.tags_to_entities(tags) == [("PER", 0, 2), ("LOC", 3, 4), ("ORG", 5, 7), ("ORG", 7, 8), ("LOC", 8, 9)]
    assert nc.tags_to_entities([]) == [] and nc.tags_to_entities(["O", "O"]) == []
    assert nc.entities_to_tags(5, [("PER", 0, 2), ("LOC", 4, 5)]) == ["B-PER", "I-PER", "O", "O", "B-LOC"]


def test_invalid_bio_transitions():
    assert nc.invalid_bio_transitions(["O", "I-PER", "I-PER", "B-ORG", "I-LOC"]) == 2
    assert nc.invalid_bio_transitions(["B-PER", "I-PER", "O"]) == 0


def test_new_york_example_token_correct_but_entity_wrong():
    gold = [["B-LOC", "I-LOC"]]
    pred = [["B-LOC", "O"]]
    tm = nc.token_metrics(gold, pred)
    assert tm["accuracy"] == 0.5 and tm["tp"] == 1 and tm["fn"] == 1            # token New dung
    em = nc.entity_metrics(gold, pred)["micro"]
    assert (em["tp"], em["fp"], em["fn"]) == (0, 1, 1) and em["f1"] == 0.0        # entity "New York" khong dung


def test_entity_metrics_hand_computed():
    gold = [["B-PER", "I-PER", "O", "B-ORG"], ["B-LOC", "O", "B-MISC", "I-MISC"]]
    pred = [["B-PER", "I-PER", "O", "B-LOC"], ["B-LOC", "O", "B-MISC", "O"]]
    m = nc.entity_metrics(gold, pred)
    # gold: PER(0,2) ORG(3,4) | LOC(0,1) MISC(2,4) ;  pred: PER(0,2) LOC(3,4) | LOC(0,1) MISC(2,3)
    assert (m["micro"]["tp"], m["micro"]["fp"], m["micro"]["fn"]) == (2, 2, 2)
    assert m["micro"]["precision"] == m["micro"]["recall"] == m["micro"]["f1"] == 0.5
    assert (m["PER"]["tp"], m["PER"]["fp"], m["PER"]["fn"]) == (1, 0, 0)
    assert (m["LOC"]["tp"], m["LOC"]["fp"], m["LOC"]["fn"]) == (1, 1, 0)
    assert (m["ORG"]["tp"], m["ORG"]["fp"], m["ORG"]["fn"]) == (0, 0, 1)
    assert (m["MISC"]["tp"], m["MISC"]["fp"], m["MISC"]["fn"]) == (0, 1, 1)


def test_entity_length_buckets_partition_entities():
    gold = [["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-PER"]]
    pred = [["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "B-PER"]]
    m = nc.entity_metrics_by_length(gold, pred)
    assert (m["1"]["tp"], m["1"]["fp"], m["1"]["fn"]) == (1, 0, 0)
    # gold ORG dai 5, prediction ORG dai 4 -> ca hai thuoc nhom 4+ nhung khac ranh gioi: 1 FP + 1 FN
    assert (m["4+"]["tp"], m["4+"]["fp"], m["4+"]["fn"]) == (0, 1, 1)
    assert (m["3"]["tp"], m["3"]["fp"], m["3"]["fn"]) == (0, 0, 0)


def test_token_accuracy_inflated_by_O():
    gold = [["O"] * 8 + ["B-PER", "I-PER"]]
    pred = [["O"] * 10]
    tm = nc.token_metrics(gold, pred)
    assert tm["accuracy"] == 0.8 and tm["f1"] == 0.0 and tm["majority_O_accuracy"] == 0.8


def test_categorize_errors():
    gold = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O", "B-MISC", "O"]
    pred = ["B-PER", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O", "B-ORG"]
    # gold: PER(0,2) ORG(3,4) LOC(5,7) MISC(8,9) ; pred: PER(0,1) LOC(3,4) LOC(5,6) ORG(9,10)
    cats = {(e["category"], e["gold"], e["pred"]) for e in nc.categorize_errors(gold, pred)}
    assert cats == {("boundary", ("PER", 0, 2), ("PER", 0, 1)), ("wrong_type", ("ORG", 3, 4), ("LOC", 3, 4)),
                    ("boundary", ("LOC", 5, 7), ("LOC", 5, 6)), ("missing", ("MISC", 8, 9), None),
                    ("spurious", None, ("ORG", 9, 10))}
    assert nc.categorize_errors(gold, gold) == []


# ---------------- Alignment ---------------- #
def test_align_labels_strategies_synthetic():
    # [CLS] play ##ing in Par ##is [SEP]   -> word_ids None,0,0,1,2,2,None
    wids = [None, 0, 0, 1, 2, 2, None]
    labels = ["B-PER", "O", "B-LOC"]
    a = nc.align_labels(wids, labels, "A")
    b = nc.align_labels(wids, labels, "B")
    L = nc.LABEL2ID
    assert a == [-100, L["B-PER"], -100, L["O"], L["B-LOC"], -100, -100]
    assert b == [-100, L["B-PER"], L["I-PER"], L["O"], L["B-LOC"], L["I-LOC"], -100]
    assert nc.align_labels(wids, ["I-ORG", "O", "O"], "B")[2] == L["I-ORG"]   # I-X giu nguyen I-X


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(nc.MODEL_NAME)


def test_alignment_with_real_tokenizer_roundtrips_word_labels(tok):
    words = ["Barack", "Obama", "visited", "Paris", ".", "Schwarzenegger"]
    tags = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-PER"]
    enc = tok(words, is_split_into_words=True)
    wids = enc.word_ids()
    lab = nc.align_labels(wids, tags, "A")
    # So nhan khac -100 phai bang so tu; thu tu nhan = thu tu tu
    kept = [l for l in lab if l != -100]
    assert [nc.LABELS[l] for l in kept] == tags
    assert wids[0] is None and wids[-1] is None and lab[0] == lab[-1] == -100
    assert len(tok.convert_ids_to_tokens(enc["input_ids"])) > len(words) + 2        # co subword
    pos = nc.first_subword_positions(wids)
    assert sorted(pos) == list(range(len(words))) and all(lab[p] != -100 for p in pos.values())


def test_truncation_drops_words_and_marks_them(tok):
    recs = [{"id": "0", "tokens": ["word"] * 50, "tags": ["O"] * 50}]
    ds = nc.NERDataset(recs, tok, "A", max_len=10)
    assert ds.n_truncated_sentences == 1 and ds.n_truncated_words == 50 - 8      # 10 - [CLS] - [SEP] = 8 tu


# ---------------- Model: shape + IGNORE ---------------- #
def test_shapes_and_ignore_index_loss():
    nc.set_seed(0)
    model = nc.TokenClassifier().eval()
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(nc.MODEL_NAME)
    ds = nc.NERDataset([{"id": "0", "tokens": ["Barack", "Obama", "visited", "Paris", "."], "tags": ["B-PER", "I-PER", "O", "B-LOC", "O"]},
                        {"id": "1", "tokens": ["Hi"], "tags": ["O"]}], t, "A", 32)
    batch = nc.make_collate(t.pad_token_id)([ds[0], ds[1]])
    with torch.no_grad():
        H = model.hidden_states(batch["input_ids"], batch["attention_mask"])
        Z = model(batch["input_ids"], batch["attention_mask"])
    B, T = batch["input_ids"].shape
    assert H.shape == (B, T, 768) and Z.shape == (B, T, 9) and batch["labels"].shape == (B, T)
    # IGNORE khong dong gop vao loss: doi logits o vi tri -100 khong doi loss; doi o vi tri co nhan thi doi
    lab = batch["labels"]
    base = nc.token_loss(Z, lab)
    Z2 = Z.clone(); Z2[lab == -100] += 5.0 * torch.randn_like(Z2[lab == -100])
    assert torch.allclose(nc.token_loss(Z2, lab), base)
    Z3 = Z.clone(); Z3[lab != -100] += 5.0 * torch.randn_like(Z3[lab != -100])
    assert not torch.allclose(nc.token_loss(Z3, lab), base)
    # loss = trung binh -log p tai cac token co nhan
    lp = torch.log_softmax(Z, -1)
    m = lab != -100
    manual = -lp[m].gather(1, lab[m].unsqueeze(1)).mean()
    assert torch.allclose(manual, base, atol=1e-5)


def test_padding_invariance_of_token_logits(tok):
    nc.set_seed(0)
    model = nc.TokenClassifier().eval()
    enc1 = tok([["Paris", "is", "nice"]], is_split_into_words=True, return_tensors="pt")
    enc2 = tok([["Paris", "is", "nice"]], is_split_into_words=True, return_tensors="pt", padding="max_length", max_length=16)
    with torch.no_grad():
        z1 = model(enc1["input_ids"], enc1["attention_mask"])
        z2 = model(enc2["input_ids"], enc2["attention_mask"])
    n = enc1["input_ids"].shape[1]
    assert torch.allclose(z1, z2[:, :n], atol=1e-4)
