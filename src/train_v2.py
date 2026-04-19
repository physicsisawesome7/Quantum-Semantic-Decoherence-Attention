"""
train_v2.py — Enhancement 5: Uncertainty-Calibrated Loss
=========================================================

Standard cross-entropy treats every prediction equally. But a model that
is confidently wrong is far more dangerous than one that is uncertainly
wrong. We add two extra terms:

  L_total = L_CE  +  λ · L_calibration  −  μ · L_entropy_bonus

L_calibration:
  Soft proxy for ECE. We want the model's softmax confidence c_i to
  match its actual accuracy. On a mini-batch, we compute:
      L_cal = |mean(correct_i) − mean(confidence_i)|
  This drives confidence ↔ accuracy alignment at the batch level.

L_entropy_bonus (rewards uncertainty on hard examples):
  When the model predicts incorrectly, we reward it for having high
  global uncertainty (high p_global). This shapes the model to say
  "I don't know" rather than confidently hallucinate.
  We only apply this to misclassified examples to avoid penalising
  correct confident predictions.

      L_ent = − E[p_global | misclassified]

Together these create a training signal that:
  - Maintains accuracy (L_CE dominates)
  - Pushes calibration (L_cal)
  - Builds epistemic humility (L_ent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from src.train import expected_calibration_error, negative_log_likelihood


# ─────────────────────────────────────────────────────────────
# Uncertainty-Calibrated Loss
# ─────────────────────────────────────────────────────────────

class UncertaintyCalibrationLoss(nn.Module):
    """
    Combined loss: CE + calibration proxy + entropy bonus.
    """
    def __init__(self, lambda_cal: float = 0.5, mu_ent: float = 0.3):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.mu_ent     = mu_ent

    def forward(
        self,
        logits:   torch.Tensor,  # (B, C)
        labels:   torch.Tensor,  # (B,)
        p_global: torch.Tensor,  # (B,) — model's uncertainty
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns (total_loss, component_dict).
        """
        # 1. Cross-entropy (primary signal)
        L_ce = F.cross_entropy(logits, labels)

        # 2. Calibration proxy
        probs      = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # (B,)
        correct    = (probs.argmax(dim=-1) == labels).float()

        # Differentiable approximation of |acc - conf|
        L_cal = (confidence - correct).abs().mean()

        # 3. Entropy bonus on misclassified examples
        misclassified = (probs.argmax(dim=-1) != labels)
        if misclassified.sum() > 0:
            # Reward higher p_global when wrong
            L_ent = -p_global[misclassified].mean()
        else:
            L_ent = torch.tensor(0.0, device=logits.device)

        # Total
        L_total = (
            L_ce
            + self.lambda_cal * L_cal
            + self.mu_ent     * L_ent
        )

        return L_total, {
            'L_ce':  L_ce.item(),
            'L_cal': L_cal.item(),
            'L_ent': L_ent.item(),
        }


# ─────────────────────────────────────────────────────────────
# Training loop for v2
# ─────────────────────────────────────────────────────────────

def train_qsda_v2(
    model,
    train_loader,
    val_loader,
    epochs:     int   = 30,
    lr:         float = 3e-4,
    device:     str   = 'cpu',
    lambda_cal: float = 0.3,
    mu_ent:     float = 0.2,
) -> Dict:
    """Train QSDA-v2 with calibration-aware loss."""
    model     = model.to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn   = UncertaintyCalibrationLoss(lambda_cal=lambda_cal, mu_ent=mu_ent)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc':    [], 'val_ece':   [], 'val_nll': [],
        'L_ce': [], 'L_cal': [], 'L_ent': [],
        'route_entropy': [],   # how much the router diversifies
        'coherence_mean': [],  # mean reasoning coherence
    }

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        comp = {'L_ce': 0.0, 'L_cal': 0.0, 'L_ent': 0.0}

        for batch in train_loader:
            ids, labels, ambiguity = [x.to(device) for x in batch]

            logits, p_global, layer_metrics = model(ids)

            loss, components = loss_fn(logits, labels, p_global)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_loss    += loss.item() * len(labels)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += len(labels)
            for k in comp: comp[k] += components[k] * len(labels)

        sched.step()

        history['train_loss'].append(t_loss / t_total)
        history['train_acc'].append(t_correct / t_total)
        for k in comp: history[k].append(comp[k] / t_total)

        val_m = evaluate_v2(model, val_loader, device)
        history['val_acc'].append(val_m['accuracy'])
        history['val_ece'].append(val_m['ece'])
        history['val_nll'].append(val_m['nll'])
        history['route_entropy'].append(val_m.get('route_entropy', 0.0))
        history['coherence_mean'].append(val_m.get('coherence_mean', 0.0))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:2d}/{epochs} | "
                f"loss={history['train_loss'][-1]:.3f} | "
                f"val_acc={val_m['accuracy']:.3f} | "
                f"ECE={val_m['ece']:.4f} | "
                f"L_ce={comp['L_ce']/t_total:.3f} "
                f"L_cal={comp['L_cal']/t_total:.3f}"
            )

    return history


@torch.no_grad()
def evaluate_v2(model, loader, device: str = 'cpu') -> Dict:
    """Evaluate QSDA-v2, collecting rich interpretability metrics."""
    model.eval()

    all_logits, all_labels, all_ambiguity = [], [], []
    all_p_global = []
    all_route_entropy = []
    all_coherence = []
    all_entropy = []

    for batch in loader:
        ids, labels, ambiguity = [x.to(device) for x in batch]
        logits, p_global, layer_metrics = model(ids)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ambiguity.append(ambiguity.cpu())
        all_p_global.append(p_global.cpu())

        for lm in layer_metrics:
            if 'route_probs' in lm:
                # Entropy of routing distribution — high = diverse, low = specialized
                rp = lm['route_probs']
                re = -(rp * (rp + 1e-8).log()).sum(dim=-1).mean().item()
                all_route_entropy.append(re)
            if 'coherence' in lm:
                all_coherence.append(lm['coherence'].abs().mean().item())
            if 'mean_entropy' in lm:
                all_entropy.append(lm['mean_entropy'].mean().item())

    logits_all    = torch.cat(all_logits)
    labels_all    = torch.cat(all_labels)
    ambiguity_all = torch.cat(all_ambiguity)
    p_global_all  = torch.cat(all_p_global)

    probs       = F.softmax(logits_all, dim=-1).numpy()
    labs        = labels_all.numpy()
    confidences = probs.max(axis=1)
    correct     = (probs.argmax(axis=1) == labs).astype(float)

    return {
        'accuracy':       float(correct.mean()),
        'ece':            expected_calibration_error(confidences, correct),
        'nll':            negative_log_likelihood(logits_all.numpy(), labs),
        'probs':          probs,
        'labels':         labs,
        'ambiguity':      ambiguity_all.numpy(),
        'p_global':       p_global_all.numpy(),
        'route_entropy':  float(np.mean(all_route_entropy)) if all_route_entropy else 0.0,
        'coherence_mean': float(np.mean(all_coherence))     if all_coherence else 0.0,
        'mean_entropy':   float(np.mean(all_entropy))       if all_entropy else 0.0,
    }
