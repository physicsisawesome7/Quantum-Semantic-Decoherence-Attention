"""
data.py
-------
Synthetic datasets for evaluating uncertainty-aware attention.

Two tasks:

Task A — Semantic Ambiguity Classification
  Tokens from 3 semantic 'clusters' (each 30 token types).
  - Clear samples: one cluster dominates → unambiguous class
  - Ambiguous samples: two clusters mixed → genuinely uncertain
  Model should be confident on clear, uncertain on ambiguous.

Task B — Long-range Dependency
  Class signal is in position 0 and position 30 of a length-40 sequence.
  Everything else is noise. Tests whether memory attention captures range.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple


VOCAB_SIZE = 100
SEQ_LEN    = 40
N_CLASSES  = 3


class AmbiguityDataset(Dataset):
    """
    Synthetic dataset with controlled ambiguity.

    For each sample:
      - class label y ∈ {0, 1, 2}
      - ambiguity level α ∈ [0, 1]
        α=0: tokens drawn purely from class cluster
        α=1: tokens drawn equally from two class clusters (maximally ambiguous)

    Token clusters:
      class 0 → tokens 0-29
      class 1 → tokens 30-59
      class 2 → tokens 60-89
      noise   → tokens 90-99
    """
    CLUSTER_SIZE = 30
    NOISE_TOKENS = list(range(90, 100))

    def __init__(self, n_samples: int = 3000, seq_len: int = SEQ_LEN, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.seq_len = seq_len

        tokens_list  = []
        labels_list  = []
        ambiguity_list = []

        # Equal samples per class
        per_class = n_samples // N_CLASSES

        for cls in range(N_CLASSES):
            other_cls = (cls + 1) % N_CLASSES
            cls_start   = cls * self.CLUSTER_SIZE
            other_start = other_cls * self.CLUSTER_SIZE

            for i in range(per_class):
                # Ambiguity level: 1/3 clear, 1/3 moderate, 1/3 high
                if i < per_class // 3:
                    alpha = rng.uniform(0.0, 0.1)   # clear
                elif i < 2 * per_class // 3:
                    alpha = rng.uniform(0.3, 0.6)   # moderate
                else:
                    alpha = rng.uniform(0.7, 0.9)   # ambiguous

                # Sample tokens
                n_own   = int((1 - alpha) * (seq_len - 3))
                n_other = int(alpha * (seq_len - 3))
                n_noise = seq_len - n_own - n_other

                own_tokens   = rng.randint(cls_start,   cls_start   + self.CLUSTER_SIZE, n_own)
                other_tokens = rng.randint(other_start, other_start + self.CLUSTER_SIZE, max(n_other, 0))
                noise_tokens = rng.choice(self.NOISE_TOKENS, max(n_noise, 0))

                toks = np.concatenate([own_tokens, other_tokens, noise_tokens])
                rng.shuffle(toks)
                toks = toks[:seq_len]

                # For highly ambiguous cases (alpha > 0.7), sometimes flip the label
                # to simulate genuine label noise / overlap
                true_label = cls
                if alpha > 0.7 and rng.random() < 0.3:
                    true_label = other_cls

                tokens_list.append(toks)
                labels_list.append(true_label)
                ambiguity_list.append(alpha)

        self.tokens    = torch.tensor(np.array(tokens_list), dtype=torch.long)
        self.labels    = torch.tensor(labels_list, dtype=torch.long)
        self.ambiguity = torch.tensor(ambiguity_list, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx], self.ambiguity[idx]


class LongRangeDataset(Dataset):
    """
    Long-range dependency task.
    Class determined by tokens at positions 0 and 35 jointly.
    All other tokens are noise.
    """
    def __init__(self, n_samples: int = 2000, seq_len: int = 40, seed: int = 99):
        super().__init__()
        rng = np.random.RandomState(seed)

        tokens_list = []
        labels_list = []

        for _ in range(n_samples):
            # Generate noise sequence
            toks = rng.randint(0, 90, seq_len)

            # Two 'key' positions determine label
            key1 = rng.randint(0, 3)   # 0, 1, or 2 → encodes first bit
            key2 = rng.randint(0, 3)   # 0, 1, or 2 → encodes second bit
            label = (key1 + key2) % N_CLASSES

            # Encode keys as special token ranges
            toks[0]  = key1 * 10          # position 0
            toks[35] = 30 + key2 * 10     # position 35

            tokens_list.append(toks)
            labels_list.append(label)

        self.tokens = torch.tensor(np.array(tokens_list), dtype=torch.long)
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        # No ambiguity in this task
        self.ambiguity = torch.zeros(n_samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx], self.ambiguity[idx]


def get_dataloaders(
    dataset:    Dataset,
    batch_size: int = 64,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
    seed:       int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset into train/val/test loaders."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
