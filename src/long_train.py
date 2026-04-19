"""
long_train.py
-------------
100-epoch training with dense behavioral tracking.

Every epoch we record:
  Core metrics:       acc, ECE, NLL, loss components
  Quantum metrics:    mean entropy, entropy std, entropy-accuracy correlation
  Routing metrics:    fraction routed to each dim (System 1 vs System 2)
  Coherence metrics:  mean |reasoning coherence|, contradiction coherence gap
  Emergence events:   epochs where phase transitions occur

Phase transitions to watch for:
  ~ epoch 5-15:  "attention crystallisation" — routing entropy drops sharply
  ~ epoch 20-40: "decoherence learning"     — entropy-accuracy correlation rises
  ~ epoch 50+:   "reasoning emergence"      — coherence divergence appears
  ~ epoch 70+:   "calibration lock-in"      — ECE stabilises

These are analogous to stages in human skill acquisition:
  Novelty → Pattern recognition → Conceptual understanding → Fluent expertise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from src.quantum_core import von_neumann_entropy, HILBERT_DIM
from src.train import expected_calibration_error


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, opt, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-5):
        self.opt = opt
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.eta_min = eta_min
        self._base_lrs = [pg['lr'] for pg in opt.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup:
            scale = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.opt.param_groups, self._base_lrs):
            pg['lr'] = base_lr * scale


def train_100_epochs(
    model,
    train_loader,
    val_loader,
    epochs:     int   = 100,
    lr:         float = 4e-4,
    device:     str   = 'cpu',
    lambda_cal: float = 0.3,
    mu_ent:     float = 0.2,
    print_every: int  = 10,
) -> Dict:
    """
    Full 100-epoch training with dense behavioral logging.
    Returns history dict with one value per epoch for every tracked quantity.
    """
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=epochs)

    history = {
        # Core
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_ece': [], 'val_nll': [],
        'lr': [],
        # Loss components
        'L_ce': [], 'L_cal': [], 'L_ent': [],
        # Quantum entropy
        'entropy_mean': [], 'entropy_std': [], 'entropy_clear': [], 'entropy_ambig': [],
        'entropy_acc_corr': [],          # Pearson r(−entropy, accuracy): should rise
        # Routing (System 1 / 2)
        'route_frac_d4': [], 'route_frac_d8': [], 'route_frac_d16': [],
        'route_specialisation': [],       # std of routing probs — higher = more specialised
        # Reasoning coherence
        'coherence_mean': [], 'coherence_std': [],
        'coherence_correct': [], 'coherence_wrong': [],
        'coherence_gap': [],              # coherence_correct - coherence_wrong (should grow)
        # Phase transitions (filled in post-hoc)
        'phase_events': {},
    }

    for epoch in range(epochs):
        sched.step(epoch)
        current_lr = opt.param_groups[0]['lr']

        # ── Train ──
        model.train()
        t_loss = t_ce = t_cal = t_ent_loss = t_correct = t_total = 0.0

        for batch in train_loader:
            ids, labels, ambiguity = [x.to(device) for x in batch]
            logits, p_global, _ = model(ids)

            L_ce  = F.cross_entropy(logits, labels)
            probs  = F.softmax(logits, dim=-1)
            conf   = probs.max(dim=-1).values
            corr   = (probs.argmax(dim=-1) == labels).float()
            L_cal  = (conf - corr).abs().mean()
            wrong  = ~corr.bool()
            L_ent  = -p_global[wrong].mean() if wrong.sum() > 0 else torch.tensor(0.0)
            loss   = L_ce + lambda_cal * L_cal + mu_ent * L_ent

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_loss    += loss.item() * len(labels)
            t_ce      += L_ce.item()  * len(labels)
            t_cal     += L_cal.item() * len(labels)
            t_ent_loss+= L_ent.item() * len(labels)
            t_correct += corr.sum().item()
            t_total   += len(labels)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(t_loss / t_total)
        history['train_acc'].append(t_correct / t_total)
        history['L_ce'].append(t_ce / t_total)
        history['L_cal'].append(t_cal / t_total)
        history['L_ent'].append(t_ent_loss / t_total)
        history['lr'].append(current_lr)

        # ── Validate with behavioral metrics ──
        val_m = _evaluate_behavioral(model, val_loader, device)
        history['val_acc'].append(val_m['accuracy'])
        history['val_ece'].append(val_m['ece'])
        history['val_nll'].append(val_m['nll'])
        history['entropy_mean'].append(val_m['entropy_mean'])
        history['entropy_std'].append(val_m['entropy_std'])
        history['entropy_clear'].append(val_m['entropy_clear'])
        history['entropy_ambig'].append(val_m['entropy_ambig'])
        history['entropy_acc_corr'].append(val_m['entropy_acc_corr'])
        history['route_frac_d4'].append(val_m['route_d4'])
        history['route_frac_d8'].append(val_m['route_d8'])
        history['route_frac_d16'].append(val_m['route_d16'])
        history['route_specialisation'].append(val_m['route_spec'])
        history['coherence_mean'].append(val_m['coh_mean'])
        history['coherence_std'].append(val_m['coh_std'])
        history['coherence_correct'].append(val_m['coh_correct'])
        history['coherence_wrong'].append(val_m['coh_wrong'])
        history['coherence_gap'].append(val_m['coh_gap'])

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(
                f"  [{epoch+1:3d}/{epochs}]  "
                f"acc={val_m['accuracy']:.3f}  ECE={val_m['ece']:.4f}  "
                f"H_mean={val_m['entropy_mean']:.3f}  "
                f"coh_gap={val_m['coh_gap']:.4f}  "
                f"route=({val_m['route_d4']:.2f}/{val_m['route_d8']:.2f}/{val_m['route_d16']:.2f})"
            )

    # Detect phase transitions
    history['phase_events'] = _detect_phases(history)
    return history


@torch.no_grad()
def _evaluate_behavioral(model, loader, device: str) -> Dict:
    model.eval()
    all_logits, all_labels, all_ambiguity = [], [], []
    all_p_global = []
    all_entropy, all_route, all_coherence = [], [], []
    coh_correct_list, coh_wrong_list = [], []

    for batch in loader:
        ids, labels, ambiguity = [x.to(device) for x in batch]
        logits, p_global, layer_metrics = model(ids)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ambiguity.append(ambiguity.cpu())
        all_p_global.append(p_global.cpu())

        # Per-sample metrics from last layer
        lm = layer_metrics[-1]

        if 'mean_entropy' in lm:
            all_entropy.append(lm['mean_entropy'].mean(dim=1).cpu())  # (B,)
        if 'route_probs' in lm:
            all_route.append(lm['route_probs'].mean(dim=1).cpu())     # (B, K)
        if 'coherence' in lm:
            coh_per_sample = lm['coherence'].abs().mean(dim=(1,2))    # (B,)
            pred_correct   = (logits.argmax(1) == labels).cpu()
            coh_correct_list.append(coh_per_sample.cpu()[pred_correct])
            coh_wrong_list.append(coh_per_sample.cpu()[~pred_correct])
            all_coherence.append(coh_per_sample.cpu())

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)
    amb_all    = torch.cat(all_ambiguity)
    probs      = F.softmax(logits_all, dim=-1).numpy()
    labs       = labels_all.numpy()
    conf       = probs.max(axis=1)
    correct    = (probs.argmax(axis=1) == labs).astype(float)

    ece = expected_calibration_error(conf, correct)
    nll = float(F.cross_entropy(logits_all, labels_all).item())

    # Entropy stats
    if all_entropy:
        ent_arr = torch.cat(all_entropy).numpy()
        amb_arr = amb_all.numpy()
        ent_mean = float(ent_arr.mean())
        ent_std  = float(ent_arr.std())
        ent_clear = float(ent_arr[amb_arr < 0.25].mean()) if (amb_arr < 0.25).sum() > 0 else ent_mean
        ent_ambig = float(ent_arr[amb_arr > 0.55].mean()) if (amb_arr > 0.55).sum() > 0 else ent_mean
        # Pearson r(-entropy, accuracy): should become more negative (entropy predicts failure)
        from scipy.stats import pearsonr
        try:
            r, _ = pearsonr(-ent_arr[:len(correct)], correct[:len(ent_arr)])
            ent_acc_corr = float(r)
        except Exception:
            ent_acc_corr = 0.0
    else:
        ent_mean = ent_std = ent_clear = ent_ambig = ent_acc_corr = 0.0

    # Routing stats
    if all_route:
        route_arr = torch.cat(all_route).numpy()   # (N, K)
        rd4  = float(route_arr[:, 0].mean())
        rd8  = float(route_arr[:, 1].mean())
        rd16 = float(route_arr[:, 2].mean())
        route_spec = float(route_arr.std(axis=1).mean())  # higher = more specialised
    else:
        rd4 = rd8 = rd16 = route_spec = 0.33

    # Coherence stats
    if all_coherence:
        coh_all = torch.cat(all_coherence).numpy()
        coh_mean = float(coh_all.mean())
        coh_std  = float(coh_all.std())
        cc = float(torch.cat(coh_correct_list).mean()) if coh_correct_list else coh_mean
        cw = float(torch.cat(coh_wrong_list).mean())   if coh_wrong_list   else coh_mean
    else:
        coh_mean = coh_std = cc = cw = 0.0
    coh_gap = cc - cw

    return {
        'accuracy': float(correct.mean()), 'ece': ece, 'nll': nll,
        'entropy_mean': ent_mean, 'entropy_std': ent_std,
        'entropy_clear': ent_clear, 'entropy_ambig': ent_ambig,
        'entropy_acc_corr': ent_acc_corr,
        'route_d4': rd4, 'route_d8': rd8, 'route_d16': rd16,
        'route_spec': route_spec,
        'coh_mean': coh_mean, 'coh_std': coh_std,
        'coh_correct': cc, 'coh_wrong': cw, 'coh_gap': coh_gap,
    }


def _detect_phases(h: Dict) -> Dict:
    """
    Detect phase transitions in training dynamics.
    Returns dict mapping phase_name → epoch_number.
    """
    events = {}
    accs = np.array(h['val_acc'])
    rspec = np.array(h['route_specialisation'])
    cgap  = np.array(h['coherence_gap'])
    eacorr = np.array(h['entropy_acc_corr'])

    # Phase 1: "Fast learning" — first epoch accuracy exceeds 60%
    above60 = np.where(accs > 0.60)[0]
    if len(above60): events['fast_learning'] = int(above60[0]) + 1

    # Phase 2: "Attention crystallisation" — routing specialisation peaks
    if len(rspec) > 5:
        peak = int(np.argmax(rspec)) + 1
        events['routing_crystallisation'] = peak

    # Phase 3: "Coherence emergence" — coherence gap first becomes positive
    pos_gap = np.where(cgap > 0.005)[0]
    if len(pos_gap): events['coherence_emergence'] = int(pos_gap[0]) + 1

    # Phase 4: "Entropy-accuracy alignment" — correlation > 0.2
    ea_thresh = np.where(eacorr > 0.20)[0]
    if len(ea_thresh): events['entropy_alignment'] = int(ea_thresh[0]) + 1

    # Phase 5: "Calibration lock-in" — ECE stabilises (std of last 10 epochs < 0.005)
    eces = np.array(h['val_ece'])
    if len(eces) >= 10 and eces[-10:].std() < 0.005:
        events['calibration_lockin'] = len(eces) - 9

    # Phase 6: "Peak accuracy" plateau
    events['peak_accuracy'] = int(np.argmax(accs)) + 1
    events['peak_accuracy_val'] = float(accs.max())

    return events
