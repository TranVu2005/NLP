"""Unit test cho qa_model.py / baseline_qa.py - gia tri tinh tay (calculation exercises cua de)."""

import math

import pandas as pd  # noqa: F401,I001  (pandas truoc torch tren Windows)
import numpy as np
import pytest
import torch

import baseline_qa as bq
import qa_model as qm

CONTEXT = "The capital of France is Paris."
QUESTION = "What is the capital of France?"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(qm.MODEL_NAME)


def _window(tok, question, context, max_len=384, stride=128, i=0):
    enc = qm.tokenize_qa(tok, [question], [context], max_len, stride)
    seq = enc.sequence_ids(i)
    offs = [o if seq[k] == 1 else (0, 0) for k, o in enumerate(enc["offset_mapping"][i])]
    return enc, seq, offs


# ------------------------------- metric ------------------------------------- #
def test_normalize_answer():
    assert qm.normalize_answer("The  City, of Paris!") == "city of paris"
    assert qm.normalize_answer("An apple") == "apple"


def test_exact_match_strict():
    assert qm.exact_match("Paris", "paris.") == 1.0
    assert qm.exact_match("the city of Paris", "Paris") == 0.0


def test_f1_paris_france_example():
    p, r, f = qm.f1_prf("France", "Paris France")            # gold = Paris France, prediction = France
    assert (p, r) == (1.0, 0.5)
    assert f == pytest.approx(2 / 3)
    # 'the city of Paris' -> normalize bo 'the' -> 'city of paris' (3 token): P = 1/3, R = 1 -> F1 = 0.5, con EM = 0
    assert qm.f1_score("the city of Paris", "Paris") == pytest.approx(0.5)


def test_f1_multiset_overlap():
    # gold 'new york city' (3), pred 'new york' (2): overlap 2 -> P = 1, R = 2/3, F1 = 0.8
    assert qm.f1_score("New York", "New York City") == pytest.approx(0.8)
    # lap tu: overlap tinh theo multiset
    assert qm.f1_score("x b b", "b b b") == pytest.approx(2 / 3)


def test_evaluate_max_over_golds():
    m = qm.evaluate_predictions(["Paris", "xx"], [["Lyon", "paris"], ["a"]])
    assert m["em"] == 0.5 and m["f1"] == 0.5


# ------------------------- alignment (muc 12, 16) ---------------------------- #
def test_character_span_exercise3(tok):
    a_s = CONTEXT.index("Paris")
    a_e = a_s + len("Paris")
    assert (a_s, a_e - 1, CONTEXT[a_s:a_e]) == (25, 29, "Paris")                  # char index (khong phai token index)
    enc, seq, offs = _window(tok, QUESTION, CONTEXT)
    sp = qm.align_span(offs, seq, a_s, a_e)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    assert toks[sp[0]] == toks[sp[1]] == "Paris"
    assert sp[0] == sp[1] == len(toks) - 3                                         # ... Paris . [SEP]  -> Paris la token thu 3 tinh tu cuoi
    assert (toks[0], toks[-1]) == ("[CLS]", "[SEP]")


def test_capital_of_france_span(tok):
    a_s = CONTEXT.index("capital of France")
    enc, seq, offs = _window(tok, QUESTION, CONTEXT)
    sp = qm.align_span(offs, seq, a_s, a_s + len("capital of France"))
    assert tok.decode(enc["input_ids"][0][sp[0]:sp[1] + 1]) == "capital of France"
    # chi so trong context (khong tinh question): [The, capital, of, France, is, Paris, .] -> start = 1, end = 3
    cs, _ = qm.context_token_range(seq)
    assert (sp[0] - cs, sp[1] - cs) == (1, 3)


def test_answer_starts_mid_token():
    offs = [(0, 0), (0, 0), (0, 4), (5, 14), (15, 20), (0, 0)]       # [CLS] q [ctx0..2] [SEP] ; ctx token 1 = 'Schwarzenegger'
    seq = [None, 0, 1, 1, 1, None]
    # answer bat dau o ky tu 9 (giua 'Schwarzenegger' 5..14) va ket thuc o ranh gioi token 20
    assert qm.align_span(offs, seq, 9, 20) == (3, 4)
    assert qm.alignment_kind(offs, seq, 9, 20) == "start_mid_token"
    assert qm.alignment_kind(offs, seq, 5, 12) == "end_mid_token"
    assert qm.alignment_kind(offs, seq, 5, 20) == "exact"
    assert qm.alignment_kind(offs, seq, 7, 12) == "both_mid_token"


def test_answer_outside_or_cut_window():
    offs = [(0, 0), (0, 0), (10, 14), (15, 20), (0, 0)]               # window chi chua context [10, 20)
    seq = [None, 0, 1, 1, None]
    assert qm.align_span(offs, seq, 0, 5) is None                      # answer nam truoc window (ngoai window)
    assert qm.align_span(offs, seq, 18, 30) is None                    # answer bi cat (end vuot qua window)
    assert qm.align_span(offs, seq, 10, 20) == (2, 3)


def test_answer_in_whitespace_gap():
    offs = [(0, 0), (0, 0), (0, 3), (5, 8), (0, 0)]
    seq = [None, 0, 1, 1, None]
    # answer_start = 4 roi vao khoang trong giua token (0,3) va (5,8) -> lay token ke can ben phai
    assert qm.align_span(offs, seq, 4, 8) == (3, 3)


# ------------------------------- loss ---------------------------------------- #
def test_start_loss_exercise2():
    p = torch.tensor([0.02, 0.05, 0.03, 0.10, 0.15, 0.60, 0.05])
    assert float(p.sum()) == pytest.approx(1.0)
    logits = p.log().unsqueeze(0)                                      # softmax(log p) = p
    start = torch.tensor([5])
    l_start = float(torch.nn.functional.cross_entropy(logits, start))
    assert l_start == pytest.approx(-math.log(0.60)) and l_start == pytest.approx(0.5108, abs=1e-4)
    # L = (L_start + L_end) / 2 voi cung phan phoi cho end
    assert float(qm.qa_loss(logits, logits, start, start)) == pytest.approx(l_start)
    # end khac: P_end(5) = 0.5 -> L = (0.5108 + 0.6931) / 2
    pe = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0 + 1e-12])
    pe = pe / pe.sum()
    assert float(qm.qa_loss(logits, pe.log().unsqueeze(0), start, start)) == pytest.approx((-math.log(0.6) - math.log(0.5)) / 2, abs=1e-4)


# ------------------------- tokenizer: stride / overflow ---------------------- #
def test_stride_overflow_windows(tok):
    ctx = " ".join(f"word{i}" for i in range(600))                    # context dai hon max_len
    enc = qm.tokenize_qa(tok, [QUESTION], [ctx], 128, 32)
    n = len(enc["input_ids"])
    assert n > 1 and set(enc["overflow_to_sample_mapping"]) == {0}
    # moi window <= max_length, chi context bi cat, hai window ke nhau overlap dung `stride` token
    assert max(len(x) for x in enc["input_ids"]) <= 128
    ctx_ids = []
    for i in range(n):
        seq = enc.sequence_ids(i)
        ctx_ids.append([t for t, s in zip(enc["input_ids"][i], seq) if s == 1])
    assert ctx_ids[0][-32:] == ctx_ids[1][:32]


def test_every_answer_in_some_window(tok):
    ctx = " ".join(f"word{i}" for i in range(600)) + " The answer is Paris. " + " ".join(f"tail{i}" for i in range(300))
    a_s = ctx.index("Paris")
    feats = qm.QAFeatures(tok, [QUESTION], [ctx], 128, 42, [(a_s, a_s + 5)])
    assert len(feats) > 3
    assert feats.n_answer_cut == 0                                      # co window chua tron answer
    assert feats.n_no_answer_windows in (len(feats) - 1, len(feats) - 2)       # answer nam trong 1 window (hoac 2 neu o vung overlap)
    pos = [(f["start_position"], f["end_position"]) for f in feats.feats if f["start_position"] > 0]
    for f in feats.feats:
        if f["start_position"] > 0:
            assert tok.decode(f["input_ids"][f["start_position"]:f["end_position"] + 1]) == "Paris"
    assert pos


def test_token_type_ids_bert_vs_dist(tok):
    enc = qm.tokenize_qa(tok, [QUESTION], [CONTEXT], 64, 16)
    assert "token_type_ids" in enc and enc.sequence_ids(0)[0] is None
    tt = enc["token_type_ids"][0]
    seq = enc.sequence_ids(0)
    assert all((t == 1) == (s == 1) or s is None for t, s in zip(tt, seq))


# ------------------------------- postprocess --------------------------------- #
def test_postprocess_constraints(tok):
    feats = qm.QAFeatures(tok, [QUESTION], [CONTEXT], 64, 16)
    f = feats.feats[0]
    T = len(f["input_ids"])
    toks = tok.convert_ids_to_tokens(f["input_ids"])
    iP = toks.index("Paris")
    iC = toks.index("capital", 1 + toks.index("[SEP]"))
    s = np.full(T, -10.0)
    e = np.full(T, -10.0)
    s[iP], e[iP] = 5.0, 5.0
    s[1], e[0] = 100.0, 100.0                                           # token cua question va [CLS]: khong duoc chon
    out = qm.postprocess(feats, [CONTEXT], [s], [e])[0]
    assert out["text"] == "Paris" and (out["start_char"], out["end_char"]) == (25, 30)
    assert out["null_score"] == pytest.approx(s[0] + e[0])
    # end < start bi loai: start cao o 'Paris', end cao o 'capital' (truoc do) -> khong chon cap (Paris, capital)
    s2, e2 = np.full(T, -10.0), np.full(T, -10.0)
    s2[iP], e2[iC] = 9.0, 9.0
    out2 = qm.postprocess(feats, [CONTEXT], [s2], [e2])[0]
    assert out2["end_tok"] >= out2["start_tok"]
    # gioi han do dai: capital .. Paris cach 5 token > max_answer_len=3 bi loai
    s3, e3 = np.full(T, -10.0), np.full(T, -10.0)
    s3[iC], e3[iP] = 9.0, 9.0
    assert qm.postprocess(feats, [CONTEXT], [s3], [e3], max_answer_len=6)[0]["text"] == "capital of France is Paris"
    assert qm.postprocess(feats, [CONTEXT], [s3], [e3], max_answer_len=3)[0]["text"] != "capital of France is Paris"


# ------------------------------- model --------------------------------------- #
def test_qamodel_shapes_and_padding_invariance(tok):
    torch.manual_seed(0)
    model = qm.QAModel(pretrained=False).eval()
    ids = torch.randint(1000, 5000, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    s, e = model(ids, mask)
    assert s.shape == e.shape == (2, 10)
    h = model.hidden_states(ids, mask)
    assert h.shape == (2, 10, 768)
    # them 4 token pad vao sample 0: logits o cac vi tri that khong doi, vi tri pad = NEG
    ids2 = torch.cat([ids[:1], torch.zeros(1, 4, dtype=torch.long)], 1)
    mask2 = torch.cat([mask[:1], torch.zeros(1, 4, dtype=torch.long)], 1)
    s2, e2 = model(ids2, mask2)
    torch.testing.assert_close(s2[0, :10], s[0], atol=1e-4, rtol=1e-4)
    assert bool((s2[0, 10:] == qm.NEG).all())


# ------------------------------- baseline ------------------------------------ #
def test_split_sentences_and_qtype():
    ctx = "Paris is big. It has many people. In 1889 the tower opened."
    sp = bq.split_sentences(ctx)
    assert [ctx[a:b] for a, b in sp] == ["Paris is big.", "It has many people.", "In 1889 the tower opened."]
    assert bq.question_type("When was it built?") == "date"
    assert bq.question_type("How many people live there?") == "number"
    assert bq.question_type("Who wrote it?") == "person"
    assert bq.question_type("Where is it?") == "place"


def test_baseline_paris_example():
    bl = bq.KeywordBaseline().fit(["Paris is the capital of France. Berlin is the capital of Germany."])
    p = bl.predict("What city is the capital of France?", "Paris is the capital of France.")
    assert p["text"] == "Paris"
    s = bl.predict("What is the capital of Germany?", "Paris is the capital of France. Berlin is the capital of Germany.")
    assert s["text"] == "Berlin"
    assert bl.predict("What is the capital of Germany?", "Paris is the capital of France. Berlin is the capital of Germany.",
                      "sentence_only")["text"] == "Berlin is the capital of Germany."
