# QSDA Mathematical Reference

This document collects every equation in the paper with derivations.

---

## 1. Mixed Quantum State

**Definition.** For a token embedding h ∈ ℝ^D, the QSDA representation is a density matrix:

```
ρ = (1 − p)|ψ⟩⟨ψ| + p · I/d
```

**Parameters:**
- `|ψ⟩ ∈ ℂ^d` — normalised pure state: `‖|ψ⟩‖ = 1`
- `p ∈ [0, 1]`  — mixing parameter (semantic uncertainty)
- `d = 2^n`     — Hilbert space dimension

**Encoder:**
```
|ψ⟩ = normalize(W_re · h + i · W_im · h)
p   = σ(MLP_p(h)) · 0.95
```

**Eigenvalues of ρ:**
```
λ₁ = 1 − p + p/d    (multiplicity 1)
λ₂ = p / d          (multiplicity d−1)
```

---

## 2. Von Neumann Entropy

**Definition.** `S(ρ) = −Tr(ρ log ρ)`

**Closed form** (from eigenvalues above):
```
S(ρ) = −λ₁ log λ₁ − (d−1) λ₂ log λ₂
```

**Boundary conditions:**
```
S(ρ)|_{p=0} = 0        (pure state, zero entropy)
S(ρ)|_{p=1} = log(d)   (maximally mixed, maximum entropy)
```

**Derivative:**
```
dS/dp = (d−1)/d · [log λ₁ − log λ₂] ≥ 0   for all p ∈ [0,1]
```

Monotonicity follows because `λ₁ ≥ λ₂` for `p ≤ 1` and `d ≥ 2`.

---

## 3. Quantum Attention (Theorem 2)

**Formula:**
```
a_{is} = Tr(ρ_i M_s) = (1−p_i)|⟨ψ_i|m_s⟩|² + p_i/d
```

where `M_s = |m_s⟩⟨m_s|` are external memory states.

**Proof:** `Tr(ρ M_s) = (1−p) Tr(|ψ⟩⟨ψ| M_s) + (p/d) Tr(I M_s)`
`= (1−p)|⟨ψ|m_s⟩|² + (p/d)·1`.

**Calibration (Theorem 2):**
```
p → 1:  a_{is} → 1/d         (uniform — maximum epistemic humility)
p → 0:  a_{is} → |⟨ψ|m_s⟩|²  (pure Born rule — certain attention)
```

---

## 4. Lindblad Decoherence (Theorem 1)

**Continuous form (Lindblad master equation):**
```
dρ/dt = Σ_k (L_k ρ L†_k − ½{L†_k L_k, ρ})
```

**Discrete approximation (one layer):**
```
p_new = p · (1 − γ)
γ = σ(MLP_γ(context)) ∈ (0, 1)
```

**Multi-layer (L layers):**
```
p_L = p_0 · ∏_{l=1}^{L} (1 − γ_l) ≤ p_0
```

**Theorem 1 (Entropy Monotonicity):** Since `S(ρ)` is monotone in `p` and each factor `(1−γ_l) ∈ [0,1]`, we have `S(ρ_L) ≤ S(ρ_0)`. □

---

## 5. Entanglement Propagation

**Semantic coupling:**
```
c_{ij} = softmax_j(h_i · h_j^T / √D)   ∈ ℝ^{N×N}
```

**Propagation update:**
```
p_i ← p_i · (1 − β · Σ_j c_{ij} · (1 − p_j))
β = σ(β_param) ∈ (0, 1)   learned coupling strength
```

**Invariant:** `p_i_new ≤ p_i` — certainty propagates from clear → uncertain, never increases. Follows since `β · Σ_j c_{ij} · (1−p_j) ≥ 0`.

---

## 6. Quantum Interference Reasoning

**Setup:** Premise states `P = {|p_s⟩}`, Conclusion states `C = {|c_s⟩}`.

**2nd-order coherence:**
```
R_{is} = Re(⟨ψ_i|p_s⟩ · ⟨p_s|c_s⟩ · ⟨c_s|ψ_i⟩)
```

**Interpretation:**
```
R_{is} > 0:  constructive interference — consistent reasoning chain
R_{is} < 0:  destructive interference — contradiction detected
|R_{is}|:    strength of the logical relationship
```

**Physical analogy:** This is the quantum version of a chain of conditional probabilities, but operating in complex Hilbert space, where sign carries meaning (phase-sensitive inference).

---

## 7. Purity

```
Tr(ρ²) = λ₁² + (d−1)λ₂²
       = (1−p+p/d)² + (d−1)(p/d)²
```

Range: `[1/d, 1]` — pure at `p=0`, maximally mixed at `p=1`.

---

## 8. Complexity Analysis (Theorem 4)

| Operation | Classical | QSDA |
|-----------|-----------|------|
| Pairwise attention | O(N²·d_k) | — |
| Memory attention | — | O(N·S·d) |
| Entanglement | — | O(N²·D) |
| QIR | — | O(N·S·d) |
| **Total per layer** | **O(N²)** | **O(N·(S+N))** |

For `S ≪ N` and ignoring entanglement: O(N·S) ≈ O(N) (linear). With entanglement: O(N²) in the worst case, but entanglement is one layer and can be made O(N·k) with sparse coupling. In practice S=16, N≤512, so QSDA is 32× cheaper than self-attention.

---

## 9. Uncertainty-Calibrated Loss

```
L_total = L_CE + λ · L_cal − μ · L_ent

L_CE  = CrossEntropy(logits, labels)
L_cal = E[|confidence − accuracy|]   (batch-level calibration proxy)
L_ent = −E[p_global | misclassified] (entropy bonus on wrong predictions)
```

**Gradient signal for `L_ent`:** On misclassified examples, the gradient increases `p_global`, making the model more uncertain. On correctly classified examples, `L_ent` is not applied, so confident-correct predictions are not penalised.

---

## 10. Training Phase Transitions

Empirically observed during 100-epoch training on the realistic benchmark:

| Epoch | Phase | Mathematical Signature |
|:-----:|-------|----------------------|
| 8 | Fast learning | `d/dt acc > 0` sharply |
| 15 | Entropy alignment | `ρ(−H(ρ), correct)` crosses 0.1 |
| 25 | Routing crystallisation | `Var(routing probs)` peaks then stabilises |
| 75 | Calibration lock-in | `std(ECE[-10:]) < 0.005` |
| 80 | Peak accuracy | `argmax acc` |

These transitions arise from the interaction of the four training objectives and are analogous to phase transitions in physical systems: sudden qualitative changes in system behaviour driven by continuous parameter changes.
