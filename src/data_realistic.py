"""
data_realistic.py
-----------------
Linguistically structured synthetic datasets that mirror real NLP tasks.

Four tasks, all tokenised from a shared 200-word vocabulary split into:
  FUNCTION  tokens  0-19   (the, a, is, was, not, very, ...)
  POSITIVE  tokens 20-59   (good, great, love, excellent, ...)
  NEGATIVE  tokens 60-99   (bad, hate, poor, terrible, ...)
  NEUTRAL   tokens 100-139 (thing, item, place, person, ...)
  DOMAIN_A  tokens 140-159 (bank, stock, trade, market, ...)   <- financial
  DOMAIN_B  tokens 160-179 (river, shore, water, fish, ...)    <- nature
  RARE      tokens 180-199 (idiosyncratic, ephemeral, ...)     <- ambiguous

Task 1 — Sentiment (3-class: pos / neg / neutral)
  Maps directly from token distribution. Tests basic classification.

Task 2 — Polarity Ambiguity (3-class: clear-pos / clear-neg / ambiguous)
  Sentences with BOTH positive and negative tokens → ambiguous class.
  Tests uncertainty quantification: ambiguous class should have high entropy.

Task 3 — Domain Word Sense Disambiguation (2-class: financial-sense / nature-sense)
  Sentence contains DOMAIN_A tokens ("bank", "stock") but the intended sense
  is determined by co-occurring context (DOMAIN_B vs more DOMAIN_A).
  Tests whether entanglement propagation resolves polysemy correctly.

Task 4 — Contradiction Detection (2-class: consistent / contradiction)
  Pair of mini-sentences. First half asserts X, second asserts not-X.
  Tests QIR: coherence should be LOW for contradictions.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from typing import Tuple, Dict

VOCAB_SIZE = 200
FUNCTION   = list(range(0,   20))
POSITIVE   = list(range(20,  60))
NEGATIVE   = list(range(60,  100))
NEUTRAL    = list(range(100, 140))
DOMAIN_A   = list(range(140, 160))   # financial
DOMAIN_B   = list(range(160, 180))   # nature
RARE       = list(range(180, 200))   # rare/ambiguous


def _rng(seed): return np.random.RandomState(seed)


class SentimentDataset(Dataset):
    """
    Task 1: Sentiment classification (0=positive, 1=negative, 2=neutral).
    Clear sentiment: one polarity dominates 70%+ of content tokens.
    """
    N_CLASSES = 3

    def __init__(self, n=1500, seq_len=32, ambiguity=0.0, seed=0):
        rng = _rng(seed)
        self.seq_len = seq_len
        toks_list, labels_list, amb_list = [], [], []

        for cls in range(self.N_CLASSES):
            per = n // self.N_CLASSES
            if   cls == 0: primary, secondary = POSITIVE, NEGATIVE
            elif cls == 1: primary, secondary = NEGATIVE, POSITIVE
            else:          primary, secondary = NEUTRAL,  POSITIVE

            for _ in range(per):
                alpha = rng.uniform(0, ambiguity) if ambiguity > 0 else rng.uniform(0, 0.15)
                n_func = rng.randint(3, 8)
                n_prim = int((seq_len - n_func) * (1 - alpha))
                n_sec  = seq_len - n_func - n_prim

                t = np.concatenate([
                    rng.choice(FUNCTION,   n_func, replace=True),
                    rng.choice(primary,    n_prim, replace=True),
                    rng.choice(secondary,  max(n_sec,1), replace=True),
                ])
                rng.shuffle(t)
                toks_list.append(t[:seq_len])
                labels_list.append(cls)
                amb_list.append(alpha)

        self.tokens    = torch.tensor(np.array(toks_list), dtype=torch.long)
        self.labels    = torch.tensor(labels_list,         dtype=torch.long)
        self.ambiguity = torch.tensor(amb_list,            dtype=torch.float)
        self.task_id   = 0

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.tokens[i], self.labels[i], self.ambiguity[i]


class PolarityAmbiguityDataset(Dataset):
    """
    Task 2: Clear-positive / Clear-negative / Genuinely-ambiguous.
    Key property: the model SHOULD have high entropy on class-2 examples.
    """
    N_CLASSES = 3

    def __init__(self, n=1500, seq_len=32, seed=1):
        rng = _rng(seed)
        toks_list, labels_list, amb_list = [], [], []
        per = n // self.N_CLASSES

        for cls in range(self.N_CLASSES):
            for _ in range(per):
                n_func = rng.randint(3, 7)

                if cls == 0:   # clear positive
                    n_pos = int((seq_len - n_func) * rng.uniform(0.75, 0.95))
                    n_neg = seq_len - n_func - n_pos
                    t = np.concatenate([rng.choice(FUNCTION, n_func, True),
                                        rng.choice(POSITIVE, n_pos,  True),
                                        rng.choice(NEGATIVE, max(n_neg,1), True)])
                    alpha = n_neg / (seq_len - n_func + 1e-6)

                elif cls == 1:  # clear negative
                    n_neg = int((seq_len - n_func) * rng.uniform(0.75, 0.95))
                    n_pos = seq_len - n_func - n_neg
                    t = np.concatenate([rng.choice(FUNCTION, n_func, True),
                                        rng.choice(NEGATIVE, n_neg,  True),
                                        rng.choice(POSITIVE, max(n_pos,1), True)])
                    alpha = n_pos / (seq_len - n_func + 1e-6)

                else:           # genuinely ambiguous: ~50/50 mix
                    half = (seq_len - n_func) // 2
                    t = np.concatenate([rng.choice(FUNCTION, n_func, True),
                                        rng.choice(POSITIVE, half, True),
                                        rng.choice(NEGATIVE, half, True)])
                    alpha = 0.5

                rng.shuffle(t)
                arr = np.zeros(seq_len, dtype=np.int64)
                arr[:min(len(t), seq_len)] = t[:seq_len]
                toks_list.append(arr)
                labels_list.append(cls)
                amb_list.append(float(alpha))

        self.tokens    = torch.tensor(np.array(toks_list), dtype=torch.long)
        self.labels    = torch.tensor(labels_list,         dtype=torch.long)
        self.ambiguity = torch.tensor(amb_list,            dtype=torch.float)
        self.task_id   = 1

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.tokens[i], self.labels[i], self.ambiguity[i]


class WSDDataset(Dataset):
    """
    Task 3: Word Sense Disambiguation.
    "bank" in financial context (label 0) vs nature context (label 1).
    All sentences contain DOMAIN_A tokens ("bank", "stock").
    Context is determined by majority of co-occurring tokens.
    Tests entanglement propagation: context resolves the ambiguous word.
    """
    N_CLASSES = 2

    def __init__(self, n=1200, seq_len=32, seed=2):
        rng = _rng(seed)
        toks_list, labels_list, amb_list = [], [], []
        per = n // self.N_CLASSES

        for cls in range(self.N_CLASSES):
            for _ in range(per):
                n_func     = rng.randint(3, 7)
                n_ambig    = rng.randint(2, 5)   # DOMAIN_A "bank" tokens
                n_context  = seq_len - n_func - n_ambig

                if cls == 0:  # financial sense: context is more DOMAIN_A
                    n_financial = int(n_context * rng.uniform(0.65, 0.90))
                    n_nature    = n_context - n_financial
                    context_tok = np.concatenate([
                        rng.choice(DOMAIN_A, n_financial, True),
                        rng.choice(DOMAIN_B, max(n_nature,1), True),
                    ])
                    alpha = n_nature / (n_context + 1e-6)
                else:          # nature sense: context is DOMAIN_B + NEUTRAL
                    n_nature    = int(n_context * rng.uniform(0.65, 0.90))
                    n_financial = n_context - n_nature
                    context_tok = np.concatenate([
                        rng.choice(DOMAIN_B, n_nature,    True),
                        rng.choice(DOMAIN_A, max(n_financial,1), True),
                    ])
                    alpha = n_financial / (n_context + 1e-6)

                t = np.concatenate([
                    rng.choice(FUNCTION,  n_func,   True),
                    rng.choice(DOMAIN_A,  n_ambig,  True),   # the ambiguous word
                    context_tok,
                ])
                rng.shuffle(t)
                toks_list.append(t[:seq_len])
                labels_list.append(cls)
                amb_list.append(float(alpha))

        self.tokens    = torch.tensor(np.array(toks_list), dtype=torch.long)
        self.labels    = torch.tensor(labels_list,         dtype=torch.long)
        self.ambiguity = torch.tensor(amb_list,            dtype=torch.float)
        self.task_id   = 2

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.tokens[i], self.labels[i], self.ambiguity[i]


class ContradictionDataset(Dataset):
    """
    Task 4: Contradiction detection.
    Consistent (label 0): both halves use same polarity tokens.
    Contradiction (label 1): first half positive, second half negative (or vice versa).
    Tests QIR reasoning coherence: contradictions should produce destructive interference.
    """
    N_CLASSES = 2

    def __init__(self, n=1200, seq_len=32, seed=3):
        rng = _rng(seed)
        toks_list, labels_list, amb_list = [], [], []
        half = seq_len // 2

        for cls in range(self.N_CLASSES):
            per = n // self.N_CLASSES
            for _ in range(per):
                n_func = rng.randint(2, 5)

                if cls == 0:  # consistent
                    pol = POSITIVE if rng.random() > 0.5 else NEGATIVE
                    h1 = rng.choice(pol, max(half - n_func//2, 1), replace=True)
                    h2 = rng.choice(pol, max(half - n_func//2, 1), replace=True)
                    alpha = 0.0
                else:         # contradiction
                    pol1 = POSITIVE if rng.random() > 0.5 else NEGATIVE
                    pol2 = NEGATIVE if pol1 is POSITIVE else POSITIVE
                    h1 = rng.choice(pol1, half - n_func//2, replace=True)
                    h2 = rng.choice(pol2, half - n_func//2, replace=True)
                    alpha = 1.0

                func_toks = rng.choice(FUNCTION, n_func, replace=True)
                t = np.concatenate([h1, func_toks, h2])
                rng.shuffle(t)
                toks_list.append(t[:seq_len])
                labels_list.append(cls)
                amb_list.append(alpha)

        self.tokens    = torch.tensor(np.array(toks_list), dtype=torch.long)
        self.labels    = torch.tensor(labels_list,         dtype=torch.long)
        self.ambiguity = torch.tensor(amb_list,            dtype=torch.float)
        self.task_id   = 3

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.tokens[i], self.labels[i], self.ambiguity[i]


def build_all_datasets(seed: int = 0):
    """Returns (train_loader, val_loader, test_loader, task_meta)."""
    datasets = [
        SentimentDataset(n=1200,      seq_len=24, seed=seed),
        PolarityAmbiguityDataset(n=1200, seq_len=24, seed=seed+1),
        WSDDataset(n=900,             seq_len=24, seed=seed+2),
        ContradictionDataset(n=900,   seq_len=24, seed=seed+3),
    ]
    combined = ConcatDataset(datasets)

    n = len(combined)
    n_tr = int(n * 0.70); n_v = int(n * 0.15); n_te = n - n_tr - n_v
    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(combined, [n_tr, n_v, n_te], generator=g)

    kw = dict(batch_size=512, num_workers=0, pin_memory=False)
    return (
        DataLoader(tr, shuffle=True,  **kw),
        DataLoader(va, shuffle=False, **kw),
        DataLoader(te, shuffle=False, **kw),
        {'n_classes_max': 3, 'n_tasks': 4, 'total': n}
    )
