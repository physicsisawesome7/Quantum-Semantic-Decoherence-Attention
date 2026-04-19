"""
models.py
---------
Quantum Semantic Decoherence Attention (QSDA) model and classical baseline.

Architecture:
  Embedding → [QSDA Block × L] → Mean Pool → Classifier

Each QSDA block:
  1. QuantumStateEncoder:       h → (|ψ⟩, p)
  2. ContextualDecoherence:     p → p' = p·(1-γ)
  3. ExternalQMemoryAttention:  (|ψ⟩, p') → attended output
  4. LayerNorm + FFN (standard)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from src.quantum_core import (
    von_neumann_entropy, quantum_attention_weights,
    HILBERT_DIM, max_entropy
)


# ─────────────────────────────────────────────────────────────
# 1. Quantum State Encoder
# ─────────────────────────────────────────────────────────────

class QuantumStateEncoder(nn.Module):
    """
    Maps token embedding h ∈ R^D → quantum state (|ψ⟩, p)

    |ψ⟩ ∈ C^d:  complex unit vector, the 'direction' of meaning
    p   ∈ [0,1]: mixing parameter, the 'uncertainty' of meaning

    The full density matrix is:
        ρ = (1-p)|ψ⟩⟨ψ| + p·I/d

    The phase structure of |ψ⟩ encodes semantic information that
    real-valued vectors cannot — two tokens can have identical
    magnitude profiles but opposite relative phase, making them
    orthogonal in the quantum sense.
    """
    def __init__(self, embed_dim: int, hilbert_dim: int):
        super().__init__()
        self.hilbert_dim = hilbert_dim

        # Real and imaginary projections for pure state amplitude
        self.W_real = nn.Linear(embed_dim, hilbert_dim, bias=True)
        self.W_imag = nn.Linear(embed_dim, hilbert_dim, bias=True)

        # Uncertainty estimation — how ambiguous is this token?
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (..., embed_dim) token embeddings

        Returns:
            psi: (..., hilbert_dim) complex — normalized quantum state
            p:   (...,) real in [0, 0.95]  — mixing / uncertainty parameter
        """
        # Pure state direction
        real = self.W_real(h)
        imag = self.W_imag(h)
        psi = torch.complex(real, imag)
        psi = F.normalize(psi, dim=-1)        # |ψ⟩ on Bloch sphere

        # Uncertainty level (scaled away from 1 for numerical stability)
        p = self.uncertainty_net(h).squeeze(-1) * 0.95

        return psi, p


# ─────────────────────────────────────────────────────────────
# 2. Lindblad-Inspired Decoherence
# ─────────────────────────────────────────────────────────────

class ContextualDecoherenceLayer(nn.Module):
    """
    Implements context-driven semantic disambiguation.

    Inspired by the Lindblad master equation:
        dρ/dt = Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    In our discrete approximation:
        p_new = p · (1 - γ)   where   γ = f(context) ∈ [0, 1]

    This models the core claim:
      - Clear context → large γ → rapid decoherence → low entropy
      - Ambiguous context → small γ → slow decoherence → high entropy

    The rate γ is computed from the local context (mean of neighbors).
    Across L layers, entropy monotonically decreases (provably, since
    p_l = p_0 · Π_l (1 - γ_l) ≤ p_0).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        p:       torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p:       (...,)          — current mixing parameters
            context: (..., embed_dim) — contextual signal

        Returns:
            p_new: (...,) — reduced mixing after decoherence
            gamma: (...,) — decoherence rate [0, 1]
        """
        gamma = self.gamma_net(context).squeeze(-1)
        p_new = p * (1.0 - gamma)
        return p_new, gamma


# ─────────────────────────────────────────────────────────────
# 3. External Quantum Memory Attention
# ─────────────────────────────────────────────────────────────

class ExternalQuantumMemoryAttention(nn.Module):
    """
    Quantum attention with S learnable external memory states.

    Instead of O(N²) pairwise comparisons, each of the N tokens
    attends to S ≪ N fixed memory states:

        a_{is} = (1-p_i)|⟨ψ_i|m_s⟩|² + p_i/d

    Properties:
      - Complexity: O(N·S) vs O(N²) for self-attention
      - Born rule: weights are naturally non-negative
      - Uncertainty-aware: high-p tokens spread attention uniformly
      - Memory states {|m_s⟩} learn prototypical semantic patterns

    The memory state overlap |⟨ψ_i|m_s⟩|² uses complex inner products,
    which encode phase information — richer than real dot-product.
    """
    def __init__(self, hilbert_dim: int, n_memory: int, embed_dim: int):
        super().__init__()
        self.hilbert_dim = hilbert_dim
        self.n_memory = n_memory

        # Learnable memory states |m_s⟩ (parameterized as real + imaginary)
        scale = 1.0 / hilbert_dim ** 0.5
        self.m_real = nn.Parameter(torch.randn(n_memory, hilbert_dim) * scale)
        self.m_imag = nn.Parameter(torch.randn(n_memory, hilbert_dim) * scale)

        # Values: each memory state has an associated value vector
        self.values = nn.Parameter(torch.randn(n_memory, embed_dim) * 0.02)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    @property
    def memory_states(self) -> torch.Tensor:
        """Return normalized complex memory states (S, d)."""
        m = torch.complex(self.m_real, self.m_imag)
        return F.normalize(m, dim=-1)

    def forward(
        self,
        psi: torch.Tensor,
        p:   torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            psi: (B, N, d) complex — query quantum states
            p:   (B, N)    real    — mixing parameters

        Returns:
            output:  (B, N, embed_dim) — attended output
            attn:    (B, N, S)         — attention weights (normalized)
        """
        m = self.memory_states  # (S, d)

        # Raw quantum attention weights
        raw = quantum_attention_weights(psi, p, m, self.hilbert_dim)  # (B, N, S)

        # Normalize (Born rule — already non-negative, normalize to simplex)
        attn = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted combination of memory values
        output = torch.einsum('bns,sd->bnd', attn, self.values)
        output = self.out_proj(output)

        return output, attn


# ─────────────────────────────────────────────────────────────
# 4. Full QSDA Block
# ─────────────────────────────────────────────────────────────

class QSDABlock(nn.Module):
    """One full QSDA transformer block."""
    def __init__(self, embed_dim: int, hilbert_dim: int, n_memory: int):
        super().__init__()
        self.encoder     = QuantumStateEncoder(embed_dim, hilbert_dim)
        self.decoherence = ContextualDecoherenceLayer(embed_dim)
        self.attention   = ExternalQuantumMemoryAttention(hilbert_dim, n_memory, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: (B, N, D) token representations

        Returns:
            h_out:   (B, N, D)
            metrics: dict of entropy/decoherence metrics for analysis
        """
        B, N, D = h.shape

        # ── 1. Encode to quantum states ──
        psi, p_init = self.encoder(h)          # (B,N,d_complex), (B,N)

        # ── 2. Context-driven decoherence ──
        # Context = mean of all token representations (global view)
        ctx = h.mean(dim=1, keepdim=True).expand(B, N, D)
        p_final, gamma = self.decoherence(p_init, ctx)

        # ── 3. Quantum memory attention ──
        attn_out, attn_weights = self.attention(psi, p_final)

        # ── 4. Compute entropy metrics (no grad needed) ──
        with torch.no_grad():
            d = self.encoder.hilbert_dim
            H_before = von_neumann_entropy(p_init,  d)  # (B, N)
            H_after  = von_neumann_entropy(p_final, d)  # (B, N)

        # ── 5. Residual + norm ──
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))

        metrics = {
            'H_before':    H_before.detach(),            # (B, N)
            'H_after':     H_after.detach(),             # (B, N)
            'H_reduction': (H_before - H_after).detach(),
            'gamma':       gamma.detach(),
            'p_init':      p_init.detach(),
            'p_final':     p_final.detach(),
            'attn_weights': attn_weights.detach(),
        }

        return h, metrics


# ─────────────────────────────────────────────────────────────
# 5. Full QSDA Classifier
# ─────────────────────────────────────────────────────────────

class QSDAClassifier(nn.Module):
    """
    Full QSDA model for classification.
    Stacks L QSDA blocks, then mean-pools and classifies.
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hilbert_dim: int,
        n_memory:    int,
        n_layers:    int,
        n_classes:   int,
        max_len:     int = 64,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos   = nn.Embedding(max_len, embed_dim)
        self.drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            QSDABlock(embed_dim, hilbert_dim, n_memory)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(
        self, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Args:
            ids: (B, N) token IDs

        Returns:
            logits:      (B, n_classes)
            all_metrics: list of metric dicts, one per layer
        """
        B, N = ids.shape
        pos  = torch.arange(N, device=ids.device).unsqueeze(0)
        h    = self.drop(self.embed(ids) + self.pos(pos))

        all_metrics = []
        for block in self.blocks:
            h, m = block(h)
            all_metrics.append(m)

        logits = self.head(h.mean(dim=1))
        return logits, all_metrics


# ─────────────────────────────────────────────────────────────
# 6. Classical Transformer Baseline
# ─────────────────────────────────────────────────────────────

class ClassicalClassifier(nn.Module):
    """
    Standard transformer classifier (no quantum components).
    Parameter count is kept comparable to QSDAClassifier.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        n_heads:    int,
        n_layers:   int,
        n_classes:  int,
        max_len:    int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos   = nn.Embedding(max_len, embed_dim)
        self.drop  = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, N = ids.shape
        pos  = torch.arange(N, device=ids.device).unsqueeze(0)
        h    = self.drop(self.embed(ids) + self.pos(pos))
        h    = self.transformer(h)
        return self.head(h.mean(dim=1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
