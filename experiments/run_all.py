"""
experiments/run_all.py
======================
Unified experiment runner. Supports three modes:

    python experiments/run_all.py --mode quick    # ~2 min, sanity check
    python experiments/run_all.py --mode full     # ~5 min, paper results
    python experiments/run_all.py --mode long     # ~15 min, 100-epoch study

Output:  results/<mode>_results.json   + PNG figures in results/
"""

import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.quantum_core   import validate_all_theorems, HILBERT_DIM, max_entropy, von_neumann_entropy_numpy
from src.models         import QSDAClassifier, ClassicalClassifier, count_params
from src.models_v2      import QSDAv2Classifier
from src.data           import AmbiguityDataset, get_dataloaders
from src.data_realistic import build_all_datasets
from src.train          import train_qsda, evaluate
from src.train_v2       import train_qsda_v2, evaluate_v2
from src.long_train     import train_100_epochs, _evaluate_behavioral

os.makedirs("results", exist_ok=True)


# ── Configs ──────────────────────────────────────────────────────────

CONFIGS = {
    "quick": dict(
        vocab_size=100, embed_dim=32, hilbert_dim=4,
        n_heads=2, n_memory_per_head=4, n_reasoning_pairs=4,
        n_layers=1, n_classes=3, max_len=24, seq_len=24,
        batch_size=256, epochs=15, lr=4e-4, n_samples=900,
        n_heads_cls=2,
    ),
    "full": dict(
        vocab_size=100, embed_dim=64, hilbert_dim=HILBERT_DIM,
        n_heads=4, n_memory_per_head=8, n_reasoning_pairs=8,
        n_layers=2, n_classes=3, max_len=40, seq_len=40,
        batch_size=128, epochs=25, lr=3e-4, n_samples=2000,
        n_heads_cls=4,
    ),
    "long": dict(
        vocab_size=200, embed_dim=32, hilbert_dim=4,
        n_heads=2, n_memory_per_head=4, n_reasoning_pairs=4,
        n_layers=1, n_classes=3, max_len=24, seq_len=24,
        batch_size=512, epochs=100, lr=4e-4, n_samples=4200,
        n_heads_cls=2,
        use_realistic=True,
    ),
}


# ── Plotting ─────────────────────────────────────────────────────────

def plot_results(results: dict, mode: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0d0d1a")
    PURPLE = "#a78bfa"; TEAL = "#34d399"; AMBER = "#fbbf24"
    BG = "#1a1a2e"; GRID = "#252540"; WHITE = "#e2e8f0"; GRAY = "#64748b"

    Q = results.get("qsda_v2", results.get("qsda_v1"))
    C = results["classical"]

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.7)

    # Accuracy
    ax = axes[0, 0]
    ep = range(1, len(Q["history"]["val_acc"]) + 1)
    ax.plot(ep, Q["history"]["val_acc"], color=PURPLE, lw=2.5, label="QSDA")
    ax.plot(range(1, len(C["history"]["val_acc"]) + 1),
            C["history"]["val_acc"], color=TEAL, lw=1.8, ls="--", label="Classical")
    ax.set_title("Validation accuracy", color=WHITE, fontsize=10)
    ax.set_xlabel("Epoch", color=GRAY, fontsize=9)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

    # ECE
    ax = axes[0, 1]
    ax.plot(ep, Q["history"]["val_ece"], color=PURPLE, lw=2.5, label="QSDA")
    ax.plot(range(1, len(C["history"]["val_ece"]) + 1),
            C["history"]["val_ece"], color=TEAL, lw=1.8, ls="--", label="Classical")
    ax.set_title("Calibration (ECE ↓)", color=WHITE, fontsize=10)
    ax.set_xlabel("Epoch", color=GRAY, fontsize=9)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

    # Final metrics bar
    ax = axes[0, 2]
    cats = ["Accuracy", "1−ECE"]
    q_v = [Q["test"]["acc"], 1 - Q["test"]["ece"]]
    c_v = [C["test"]["acc"], 1 - C["test"]["ece"]]
    x = np.arange(2)
    ax.bar(x - 0.2, q_v, 0.35, color=PURPLE, alpha=0.85, label="QSDA")
    ax.bar(x + 0.2, c_v, 0.35, color=TEAL,   alpha=0.85, label="Classical")
    ax.set_xticks(x); ax.set_xticklabels(cats, color=GRAY, fontsize=9)
    ax.set_title("Final test metrics", color=WHITE, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

    # Entropy theory curve
    ax = axes[1, 0]
    p_vals = np.linspace(0, 1, 200)
    for d, col, lbl in [(4, TEAL, "d=4"), (8, PURPLE, "d=8"), (16, AMBER, "d=16")]:
        ax.plot(p_vals, von_neumann_entropy_numpy(p_vals, d), color=col, lw=2, label=lbl)
    ax.set_title("S(ρ) vs mixing param p", color=WHITE, fontsize=10)
    ax.set_xlabel("p", color=GRAY, fontsize=9)
    ax.set_ylabel("Entropy (nats)", color=GRAY, fontsize=9)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

    # Loss
    ax = axes[1, 1]
    ax.plot(ep, Q["history"]["train_loss"], color=PURPLE, lw=2, label="QSDA")
    ax.plot(range(1, len(C["history"]["train_loss"]) + 1),
            C["history"]["train_loss"], color=TEAL, lw=1.8, ls="--", label="Classical")
    ax.set_title("Training loss", color=WHITE, fontsize=10)
    ax.set_xlabel("Epoch", color=GRAY, fontsize=9)
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)

    # Summary text
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"Mode: {mode}",
        "",
        f"QSDA   acc = {Q['test']['acc']:.4f}",
        f"       ECE = {Q['test']['ece']:.4f}",
        f"       NLL = {Q['test']['nll']:.4f}",
        f"    params = {Q['n_params']:,}",
        "",
        f"Classic acc = {C['test']['acc']:.4f}",
        f"        ECE = {C['test']['ece']:.4f}",
        f"        NLL = {C['test']['nll']:.4f}",
        f"     params = {C['n_params']:,}",
        "",
        f"Δ acc = {(Q['test']['acc'] - C['test']['acc'])*100:+.2f}%",
    ]
    for i, ln in enumerate(lines):
        ax.text(0.05, 0.95 - i * 0.07, ln,
                color=WHITE if ln.startswith(("QSDA", "Classic", "Δ", "Mode")) else GRAY,
                fontsize=9, transform=ax.transAxes, family="monospace")

    fig.suptitle(f"QSDA Results — {mode} mode", color=WHITE, fontsize=13, y=0.98)
    path = f"results/{mode}_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Figure saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QSDA Experiment Runner")
    parser.add_argument("--mode", choices=["quick", "full", "long"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg  = CONFIGS[args.mode]
    mode = args.mode

    print("=" * 60)
    print(f"QSDA Experiment Runner  —  mode: {mode}")
    print("=" * 60)

    # ── Step 1: Validate mathematics ──────────────────────────────────
    print("\n[1/4] Validating quantum mathematics …")
    math_results = validate_all_theorems(cfg["hilbert_dim"])
    for claim, passed in math_results.items():
        print(f"  {'✓' if passed else '✗'}  {claim}")
    assert all(math_results.values()), "Math validation failed"
    print("  All theorems validated.\n")

    # ── Step 2: Build datasets ─────────────────────────────────────────
    print("[2/4] Building datasets …")
    if cfg.get("use_realistic"):
        train_l, val_l, test_l, meta = build_all_datasets(seed=args.seed)
        print(f"  Multi-task realistic benchmark: {meta['total']} samples, {meta['n_tasks']} tasks")
    else:
        ds = AmbiguityDataset(n_samples=cfg["n_samples"], seq_len=cfg["seq_len"], seed=args.seed)
        train_l, val_l, test_l = get_dataloaders(ds, batch_size=cfg["batch_size"], seed=args.seed)
        print(f"  Ambiguity dataset: {cfg['n_samples']} samples")

    # ── Step 3: Train models ──────────────────────────────────────────
    print("\n[3/4] Training models …")

    # QSDA-v2
    qv2 = QSDAv2Classifier(
        vocab_size         = cfg["vocab_size"],
        embed_dim          = cfg["embed_dim"],
        hilbert_dim        = cfg["hilbert_dim"],
        n_heads            = cfg["n_heads"],
        n_memory_per_head  = cfg["n_memory_per_head"],
        n_reasoning_pairs  = cfg["n_reasoning_pairs"],
        n_layers           = cfg["n_layers"],
        n_classes          = cfg["n_classes"],
        max_len            = cfg["max_len"],
    )
    classical = ClassicalClassifier(
        vocab_size = cfg["vocab_size"],
        embed_dim  = cfg["embed_dim"],
        n_heads    = cfg["n_heads_cls"],
        n_layers   = cfg["n_layers"],
        n_classes  = cfg["n_classes"],
        max_len    = cfg["max_len"],
    )
    print(f"  QSDA-v2 params:   {count_params(qv2):,}")
    print(f"  Classical params: {count_params(classical):,}")

    print("\n  QSDA-v2:")
    t0 = time.time()
    if mode == "long":
        qh = train_100_epochs(qv2, train_l, val_l, epochs=cfg["epochs"],
                               lr=cfg["lr"], print_every=20)
    else:
        qh = train_qsda_v2(qv2, train_l, val_l, epochs=cfg["epochs"],
                            lr=cfg["lr"])
    print(f"  Trained in {time.time()-t0:.1f}s")

    print("\n  Classical:")
    t0 = time.time()
    ch = train_qsda(classical, train_l, val_l, epochs=cfg["epochs"],
                    lr=cfg["lr"], is_qsda=False)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # ── Step 4: Evaluate and save ─────────────────────────────────────
    print("\n[4/4] Evaluating …")
    if mode == "long":
        qt = _evaluate_behavioral(qv2, test_l, "cpu")
    else:
        qt = evaluate_v2(qv2, test_l, "cpu")
    ct = evaluate(classical, test_l, "cpu", is_qsda=False)

    print(f"\n  QSDA-v2:   acc={qt['accuracy']:.4f}  ECE={qt['ece']:.4f}  NLL={qt['nll']:.4f}")
    print(f"  Classical: acc={ct['accuracy']:.4f}  ECE={ct['ece']:.4f}  NLL={ct['nll']:.4f}")
    print(f"  Δ acc: {(qt['accuracy']-ct['accuracy'])*100:+.2f}%")

    results = {
        "mode":    mode,
        "seed":    args.seed,
        "config":  {k: v for k, v in cfg.items() if not callable(v)},
        "qsda_v2": {
            "test":    {k: float(v) for k, v in qt.items() if not hasattr(v, "__len__")},
            "history": {k: [float(x) for x in v] if isinstance(v, list) else v
                        for k, v in qh.items() if k not in ("phase_events",)},
            "phase_events": qh.get("phase_events", {}),
            "n_params": count_params(qv2),
        },
        "classical": {
            "test":    {"acc": float(ct["accuracy"]), "ece": float(ct["ece"]), "nll": float(ct["nll"])},
            "history": {"val_acc": ch["val_acc"], "val_ece": ch["val_ece"], "train_loss": ch["train_loss"]},
            "n_params": count_params(classical),
        },
        "math_validation": math_results,
    }

    out = f"results/{mode}_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")

    plot_results(results, mode)
    print("\nDone.")


if __name__ == "__main__":
    main()
