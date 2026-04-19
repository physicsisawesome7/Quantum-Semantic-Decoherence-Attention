# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/).

---

## [2.0.0] — QSDA-v2 — 2025-04

### Added
- **Multi-Head Quantum Attention (MHQA):** H parallel quantum heads with independent basis rotations and memory banks. Outputs combined via learned interference weights.
- **Entanglement Propagation Layer:** Cross-token uncertainty coupling; certainty flows from confident neighbours to ambiguous tokens.
- **Quantum Interference Reasoning (QIR):** 2nd-order coherence between premise and conclusion memory banks. Detects logical consistency and contradiction.
- **Adaptive Hilbert Space Router:** Token-wise routing to d ∈ {4, 8, 16} subspaces, implementing dual-process cognition (System 1/2).
- **Uncertainty-Calibrated Loss:** `L_CE + λ·L_cal − μ·L_ent` trains both accuracy and epistemic humility simultaneously.
- **Four-task realistic benchmark:** Sentiment, Polarity Ambiguity, WSD, Contradiction Detection over a 200-token linguistically structured vocabulary.
- **100-epoch training loop** with phase transition detection (5 phases characterised).
- **Behavioral analysis script** generating all paper figures.
- **22-test unit test suite** validating all four theorems numerically.
- `experiments/run_all.py` — unified runner with `--mode quick/full/long`.

### Changed
- `models_v2.py` replaces single-head `QSDABlock` from v1.
- `train_v2.py` adds `UncertaintyCalibrationLoss` replacing plain `CrossEntropyLoss`.
- `long_train.py` adds `WarmupCosineScheduler` replacing `CosineAnnealingLR`.

### Results
- +7.0% accuracy over classical baseline (100-epoch multi-task)
- +10.3% on polarity ambiguity, +9.9% on contradiction detection
- Spontaneous dual-process routing emergence at epoch 25
- Von Neumann entropy predicts errors (r = 0.31) from epoch 15

---

## [1.0.0] — QSDA-v1 — 2025-03

### Added
- Mixed quantum state token representation: `ρ = (1-p)|ψ⟩⟨ψ| + p·I/d`
- Analytic von Neumann entropy: `S(ρ) = −λ₁ log λ₁ − (d−1)λ₂ log λ₂`
- Single-head external quantum memory attention: `a_{is} = (1-p)|⟨ψ|m_s⟩|² + p/d`
- Lindblad decoherence layer: `p_new = p·(1−γ)` with contextual rate
- Quantum State Encoder mapping `h ∈ ℝ^D → (|ψ⟩, p)`
- Classical Transformer baseline for comparison
- Synthetic Ambiguity Dataset with controlled α parameter
- Four proved theorems: entropy monotonicity, calibration, phase expressivity, complexity
- Mathematical validation suite (8 claims)

### Results
- 84.7% accuracy on ambiguity benchmark vs 88.0% classical
- ECE 0.037 vs 0.048 classical (22% better calibration)
- Entropy-accuracy correlation demonstrated
