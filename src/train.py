"""
train.py
--------
Training loop, evaluation, and calibration metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.quantum_core import von_neumann_entropy, HILBERT_DIM


# ─────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    n_bins:      int = 10,
) -> float:
    """
    ECE = Σ_b (|B_b| / n) · |acc(B_b) - conf(B_b)|

    Lower is better. Perfect calibration = 0.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    t = torch.tensor(logits)
    y = torch.tensor(labels)
    return F.cross_entropy(t, y).item()


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_qsda(
    model,
    train_loader,
    val_loader,
    epochs:    int = 30,
    lr:        float = 3e-4,
    device:    str = 'cpu',
    is_qsda:   bool = True,
) -> Dict:
    """Train a QSDA or classical model, return history."""
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_ece':    [], 'val_nll':   [],
        'entropy_by_layer': [],   # per epoch, per layer
        'entropy_clear':    [],   # mean entropy for clear samples
        'entropy_ambig':    [],   # mean entropy for ambiguous samples
    }

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch in train_loader:
            ids, labels, ambiguity = [x.to(device) for x in batch]

            if is_qsda:
                logits, _ = model(ids)
            else:
                logits = model(ids)

            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_loss    += loss.item() * len(labels)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += len(labels)

        sched.step()

        history['train_loss'].append(t_loss / t_total)
        history['train_acc'].append(t_correct / t_total)

        # ── Validate ──
        val_metrics = evaluate(model, val_loader, device, is_qsda)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_ece'].append(val_metrics['ece'])
        history['val_nll'].append(val_metrics['nll'])

        if is_qsda and 'entropy_by_layer' in val_metrics:
            history['entropy_by_layer'].append(val_metrics['entropy_by_layer'])
            history['entropy_clear'].append(val_metrics['entropy_clear'])
            history['entropy_ambig'].append(val_metrics['entropy_ambig'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:2d}/{epochs} | "
                f"loss={history['train_loss'][-1]:.3f} | "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"ECE={val_metrics['ece']:.4f}"
            )

    return history


@torch.no_grad()
def evaluate(
    model,
    loader,
    device:  str = 'cpu',
    is_qsda: bool = True,
) -> Dict:
    """Full evaluation with calibration metrics."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_logits, all_labels, all_ambiguity = [], [], []
    all_layer_entropies = []   # list of tensors (B, N) per batch per layer
    all_layer_H_clear   = []
    all_layer_H_ambig   = []

    for batch in loader:
        ids, labels, ambiguity = [x.to(device) for x in batch]

        if is_qsda:
            logits, layer_metrics = model(ids)
            # Collect per-layer entropy
            batch_entropies = []
            for lm in layer_metrics:
                H = lm['H_after'].mean(dim=1)  # (B,) mean over sequence
                batch_entropies.append(H.cpu())

                # Separate by ambiguity
                clear_mask = ambiguity < 0.3
                ambig_mask = ambiguity > 0.6
                if clear_mask.sum() > 0:
                    all_layer_H_clear.append(H[clear_mask].mean().item())
                if ambig_mask.sum() > 0:
                    all_layer_H_ambig.append(H[ambig_mask].mean().item())

            all_layer_entropies.append(batch_entropies)
        else:
            logits = model(ids)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ambiguity.append(ambiguity.cpu())

    logits_all    = torch.cat(all_logits)
    labels_all    = torch.cat(all_labels)
    ambiguity_all = torch.cat(all_ambiguity)

    probs = F.softmax(logits_all, dim=-1).numpy()
    labs  = labels_all.numpy()

    confidences = probs.max(axis=1)
    correct     = (probs.argmax(axis=1) == labs).astype(float)

    result = {
        'loss':     loss_fn(logits_all, labels_all).item(),
        'accuracy': correct.mean(),
        'ece':      expected_calibration_error(confidences, correct),
        'nll':      negative_log_likelihood(logits_all.numpy(), labs),
        'probs':    probs,
        'labels':   labs,
        'ambiguity': ambiguity_all.numpy(),
    }

    if is_qsda and all_layer_entropies:
        # Average entropy per layer across all batches
        n_layers = len(all_layer_entropies[0])
        layer_means = []
        for l in range(n_layers):
            layer_H = np.mean([batch[l].mean().item() for batch in all_layer_entropies])
            layer_means.append(layer_H)
        result['entropy_by_layer'] = layer_means
        result['entropy_clear']    = float(np.mean(all_layer_H_clear)) if all_layer_H_clear else 0.0
        result['entropy_ambig']    = float(np.mean(all_layer_H_ambig)) if all_layer_H_ambig else 0.0

    return result


@torch.no_grad()
def collect_entropy_by_ambiguity(
    model, loader, device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect final-layer entropy vs ambiguity level.
    Returns (ambiguity, entropy, correct) arrays.
    """
    model.eval()
    all_amb, all_ent, all_cor = [], [], []

    for batch in loader:
        ids, labels, ambiguity = [x.to(device) for x in batch]
        logits, layer_metrics = model(ids)

        # Final layer entropy
        final_H = layer_metrics[-1]['H_after'].mean(dim=1)  # (B,)
        correct  = (logits.argmax(1) == labels).float()

        all_amb.append(ambiguity.cpu().numpy())
        all_ent.append(final_H.cpu().numpy())
        all_cor.append(correct.cpu().numpy())

    return (
        np.concatenate(all_amb),
        np.concatenate(all_ent),
        np.concatenate(all_cor),
    )
