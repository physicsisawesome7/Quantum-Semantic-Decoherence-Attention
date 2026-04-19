"""
experiments/analyze_behavior.py
================================
Deep behavioral analysis of a trained QSDA-v2 model.
Generates all figures used in the paper.

Usage:
    python experiments/analyze_behavior.py --results results/long_results.json
    python experiments/analyze_behavior.py --results results/full_results.json
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.quantum_core import von_neumann_entropy_numpy, max_entropy, HILBERT_DIM

PURPLE = "#c4b5fd"; TEAL = "#5eead4"; AMBER = "#fde68a"; CORAL = "#fca5a5"
GRAY = "#64748b"; BG = "#12122a"; GRID = "#1e1e3f"; WHITE = "#f8fafc"


def ax_style(ax, title, xl="", yl=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight="medium", pad=7)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.8)
    if xl: ax.set_xlabel(xl, color=GRAY, fontsize=8)
    if yl: ax.set_ylabel(yl, color=GRAY, fontsize=8)


def plot_entropy_theory(ax):
    """Panel: theoretical entropy curves for d=4,8,16."""
    p = np.linspace(0, 1, 300)
    for d, col, lbl in [(4, CORAL, "d=4"), (8, PURPLE, "d=8"), (16, TEAL, "d=16")]:
        ax.plot(p, von_neumann_entropy_numpy(p, d), color=col, lw=2, label=lbl)
    ax.fill_between(p[:160],
                    von_neumann_entropy_numpy(p[:160], 4),
                    von_neumann_entropy_numpy(p[:160], 16),
                    alpha=0.08, color=TEAL)
    ax.annotate("Adaptive routing\ngain region",
                xy=(0.15, 0.7), xytext=(0.3, 1.1),
                color=TEAL, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))
    ax_style(ax, "Theoretical: S(ρ) vs mixing param p", "p", "Entropy (nats)")
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)


def plot_entropy_collapse(ax, history):
    """Panel: entropy evolution over training."""
    eps = range(1, len(history["entropy_mean"]) + 1)
    ax.plot(eps, history["entropy_mean"], color=CORAL, lw=2.5, label="Mean S(ρ)")
    ax.fill_between(eps, history["entropy_mean"], alpha=0.2, color=CORAL)
    ax.axhline(np.log(4), color=AMBER, lw=1, ls="--",
               label=f"Max entropy log(d)={np.log(4):.2f}")
    ax_style(ax, "VN entropy collapse across training", "Epoch", "S(ρ) nats")
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=8)


def plot_routing(ax, history):
    """Panel: adaptive Hilbert routing evolution."""
    eps = range(1, len(history["route_frac_d4"]) + 1)
    ax.stackplot(eps,
                 history["route_frac_d4"],
                 history["route_frac_d8"],
                 history["route_frac_d16"],
                 colors=[TEAL, PURPLE, CORAL], alpha=0.75,
                 labels=["d=4 (System 1)", "d=8 (default)", "d=16 (System 2)"])
    ax.set_ylim(0, 1)
    ax_style(ax, "Adaptive routing: dual-process emergence", "Epoch", "Fraction")
    ax.legend(facecolor=BG, labelcolor=WHITE, fontsize=7)


def plot_coherence_gap(ax, history):
    """Panel: QIR coherence gap (correct vs wrong)."""
    eps = range(1, len(history["coherence_gap"]) + 1)
    coh = np.array(history["coherence_gap"])
    colors = [TEAL if v >= 0 else CORAL for v in coh]
    ax.bar(eps, coh, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color=GRAY, lw=1)
    ax_style(ax, "QIR coherence: correct − wrong", "Epoch", "Coherence gap")


def plot_accuracy_gap(ax, q_hist, c_hist):
    """Panel: accuracy delta QSDA - Classical over epochs."""
    n = min(len(q_hist["val_acc"]), len(c_hist["val_acc"]))
    q = np.array(q_hist["val_acc"][:n])
    c = np.array(c_hist["val_acc"][:n])
    delta = (q - c) * 100
    ax.plot(range(1, n + 1), delta, color="#f9a8d4", lw=2)
    ax.fill_between(range(1, n + 1), delta, 0,
                    where=q > c, alpha=0.2, color="#f9a8d4")
    ax.axhline(0, color=GRAY, lw=1, ls=":")
    ax.set_title("QSDA accuracy lead (%)", color=WHITE, fontsize=10, pad=7)
    ax.set_xlabel("Epoch", color=GRAY, fontsize=8)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.8)
    ax.tick_params(colors=GRAY, labelsize=8)
    if len(delta):
        ax.text(n - 1, delta[-1] + 0.3, f"+{delta[-1]:.1f}%",
                color="#f9a8d4", fontsize=10, ha="right")


def plot_phase_events(ax, phase_events):
    """Panel: phase transition timeline."""
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title("Training phase transitions", color=WHITE, fontsize=10,
                 fontweight="medium", pad=7)
    phase_meta = [
        ("fast_learning",           "#22d3ee", "Fast learning",           "acc > 60%"),
        ("routing_crystallisation", "#a78bfa", "Routing crystallises",    "specialisation peaks"),
        ("entropy_alignment",       "#4ade80", "Entropy alignment",       "H predicts errors"),
        ("calibration_lockin",      "#fb923c", "Calibration lock-in",     "ECE stabilises"),
        ("peak_accuracy",           "#f43f5e", "Peak accuracy",           "plateau reached"),
    ]
    for i, (key, col, name, desc) in enumerate(phase_meta):
        ep  = phase_events.get(key)
        y   = 0.85 - i * 0.18
        ep_str = str(ep) if ep is not None else "n/a"
        ax.add_patch(plt.Circle((0.07, y), 0.028, color=col,
                                transform=ax.transAxes, zorder=3))
        ax.text(0.15, y + 0.02, f"ep {ep_str}: {name}",
                color=col, fontsize=8.5, fontweight="medium",
                transform=ax.transAxes, va="center")
        ax.text(0.15, y - 0.04, desc,
                color=GRAY, fontsize=7.5, transform=ax.transAxes, va="center")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/long_results.json")
    parser.add_argument("--out",     default="results/behavior_analysis.png")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run experiments/run_all.py --mode long first.")
        sys.exit(1)

    with open(args.results) as f:
        R = json.load(f)

    Q = R.get("qsda_v2", R.get("qsda"))
    C = R["classical"]
    qh = Q["history"]
    ch = C["history"]
    phases = Q.get("phase_events", {})

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#080814")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_entropy_theory (fig.add_subplot(gs[0, 0]))
    plot_entropy_collapse(fig.add_subplot(gs[0, 1]), qh)
    plot_routing        (fig.add_subplot(gs[0, 2]), qh)
    plot_coherence_gap  (fig.add_subplot(gs[1, 0]), qh)
    plot_accuracy_gap   (fig.add_subplot(gs[1, 1]), qh, ch)
    plot_phase_events   (fig.add_subplot(gs[1, 2]), phases)

    q_acc = Q["test"]["acc"]; c_acc = C["test"]["acc"]
    fig.suptitle(
        f"QSDA-v2 Behavioral Analysis  |  "
        f"QSDA {q_acc:.3f}  Classical {c_acc:.3f}  "
        f"(Δ = {(q_acc - c_acc)*100:+.1f}%)",
        color=WHITE, fontsize=13, y=0.98
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
