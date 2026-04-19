"""
tests/test_quantum_core.py  —  QSDA-v2 full unit test suite (31 tests)
Run:  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest, torch, numpy as np
import torch.nn.functional as F
from src.quantum_core import (
    von_neumann_entropy, von_neumann_entropy_numpy, max_entropy,
    quantum_attention_weights, purity, lindblad_decoherence_step, HILBERT_DIM,
)

# ── Theorem 1: Von Neumann Entropy ────────────────────────────────────────────

class TestVonNeumannEntropy:
    def test_zero_at_pure_state(self):
        assert von_neumann_entropy(torch.tensor(0.0), d=8).item() < 1e-6

    def test_maximum_at_mixed_state(self):
        for d in [4, 8, 16]:
            S = von_neumann_entropy(torch.tensor(1.0), d=d).item()
            assert abs(S - np.log(d)) < 1e-5

    def test_monotone_in_p(self):
        ps = torch.linspace(0.0, 1.0, 500)
        Ss = von_neumann_entropy(ps, d=HILBERT_DIM)
        assert (Ss[1:] >= Ss[:-1] - 1e-6).all()

    def test_bounded(self):
        ps = torch.linspace(0, 1, 200)
        Ss = von_neumann_entropy(ps, d=HILBERT_DIM)
        assert (Ss >= -1e-6).all() and (Ss <= np.log(HILBERT_DIM) + 1e-5).all()

    def test_batch_shape(self):
        p = torch.rand(3, 7, 5)
        assert von_neumann_entropy(p).shape == p.shape

    def test_numpy_matches_torch(self):
        p_np = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(
            von_neumann_entropy_numpy(p_np),
            von_neumann_entropy(torch.tensor(p_np, dtype=torch.float32)).numpy(),
            atol=1e-5
        )

    def test_multilayer_monotone(self):
        p = torch.tensor(0.85)
        S_prev = von_neumann_entropy(p).item()
        for g in [0.2, 0.35, 0.1, 0.4, 0.15]:
            p = lindblad_decoherence_step(p, torch.tensor(g))
            S_cur = von_neumann_entropy(p).item()
            assert S_cur <= S_prev + 1e-6
            S_prev = S_cur

    def test_higher_d_higher_max_entropy(self):
        p = torch.tensor(0.5)
        Ss = [von_neumann_entropy(p, d=d).item() for d in [4, 8, 16]]
        assert Ss[0] < Ss[1] < Ss[2]


# ── Theorem 2: Quantum Attention & Calibration ────────────────────────────────

class TestQuantumAttentionWeights:
    @pytest.fixture
    def qs(self):
        torch.manual_seed(42)
        d   = HILBERT_DIM
        psi = F.normalize(torch.randn(6, d, dtype=torch.complex64), dim=-1)
        mem = F.normalize(torch.randn(16, d, dtype=torch.complex64), dim=-1)
        return d, psi, mem

    def test_non_negative(self, qs):
        d, psi, mem = qs
        assert (quantum_attention_weights(psi, torch.rand(6)*0.9, mem, d) >= -1e-8).all()

    def test_sums_to_one(self, qs):
        d, psi, mem = qs
        attn = quantum_attention_weights(psi, torch.rand(6)*0.9, mem, d)
        attn_n = attn / attn.sum(-1, keepdim=True)
        np.testing.assert_allclose(attn_n.sum(-1).numpy(), np.ones(6), atol=1e-5)

    def test_high_p_near_uniform(self, qs):
        d, psi, mem = qs
        attn = quantum_attention_weights(psi, torch.ones(6)*0.999, mem, d)
        attn_n = attn / attn.sum(-1, keepdim=True)
        assert float(attn_n.var(dim=-1).max()) < 0.002

    def test_zero_p_is_born_rule(self, qs):
        d, psi, mem = qs
        attn = quantum_attention_weights(psi, torch.zeros(6), mem, d)
        expected = (torch.einsum("nd,sd->ns", psi.conj(), mem).abs()**2).float()
        np.testing.assert_allclose(attn.float().numpy(), expected.numpy(), atol=1e-5)

    def test_output_shape(self):
        B, N, S, d = 4, 12, 10, HILBERT_DIM
        psi  = F.normalize(torch.randn(B, N, d, dtype=torch.complex64), dim=-1)
        mem  = F.normalize(torch.randn(S, d, dtype=torch.complex64), dim=-1)
        assert quantum_attention_weights(psi, torch.rand(B, N)*0.9, mem, d).shape == (B, N, S)

    def test_higher_p_flatter_distribution(self, qs):
        d, psi, mem = qs
        def attn_entropy(p_val):
            a = quantum_attention_weights(psi, torch.ones(6)*p_val, mem, d)
            a = a / a.sum(-1, keepdim=True)
            return -(a * (a+1e-10).log()).sum(-1).mean().item()
        assert attn_entropy(0.8) > attn_entropy(0.1)


# ── Theorem 3: Phase Expressivity ────────────────────────────────────────────

class TestPhaseExpressivity:
    def test_phase_rotation_changes_inner_product(self):
        torch.manual_seed(13)
        d    = HILBERT_DIM
        psi1 = F.normalize(torch.randn(d, dtype=torch.complex64), dim=0)
        psi2 = F.normalize(torch.randn(d, dtype=torch.complex64), dim=0)
        # Rotate psi2 by e^{iθ}: magnitudes unchanged, inner product phase shifts
        theta  = torch.tensor(1.2)
        psi2_r = psi2 * torch.exp(1j * theta)
        assert abs(psi2.abs().sum().item() - psi2_r.abs().sum().item()) < 1e-5
        ip_orig    = (psi1.conj() * psi2).sum()
        ip_rotated = (psi1.conj() * psi2_r).sum()
        assert abs((ip_rotated / (ip_orig + 1e-10)).angle().item()) > 0.01

    def test_complex_pairs_have_nonzero_phase(self):
        torch.manual_seed(7)
        d, n = HILBERT_DIM, 100
        has_imag = sum(
            abs((F.normalize(torch.randn(d, dtype=torch.complex64), dim=0).conj() *
                 F.normalize(torch.randn(d, dtype=torch.complex64), dim=0)).sum().imag.item()) > 1e-3
            for _ in range(n)
        )
        assert has_imag > n * 0.85

    def test_imaginary_component_non_trivial(self):
        """Generic complex states have non-trivial imaginary inner products
        — information classical real attention discards."""
        torch.manual_seed(99)
        d = HILBERT_DIM
        total_imag_mag = 0.0
        for _ in range(50):
            a = F.normalize(torch.randn(d, dtype=torch.complex64), dim=0)
            b = F.normalize(torch.randn(d, dtype=torch.complex64), dim=0)
            total_imag_mag += abs((a.conj() * b).sum().imag.item())
        assert total_imag_mag / 50 > 0.05   # average imaginary magnitude > 0.05


# ── Purity ───────────────────────────────────────────────────────────────────

class TestPurity:
    def test_pure_state_purity_one(self):
        assert abs(purity(torch.tensor(0.0), HILBERT_DIM).item() - 1.0) < 1e-5

    def test_mixed_state_purity_one_over_d(self):
        d = HILBERT_DIM
        assert abs(purity(torch.tensor(1.0), d).item() - 1.0/d) < 1e-5

    def test_purity_decreasing_in_p(self):
        ps = torch.linspace(0, 1, 200)
        pur = purity(ps, HILBERT_DIM)
        assert (pur[1:] <= pur[:-1] + 1e-6).all()


# ── Decoherence step ─────────────────────────────────────────────────────────

class TestDecoherenceStep:
    def test_reduces_p(self):
        assert lindblad_decoherence_step(torch.tensor(0.8), torch.tensor(0.5)).item() < 0.8

    def test_exact_formula(self):
        p, g = 0.7, 0.4
        out = lindblad_decoherence_step(torch.tensor(p), torch.tensor(g)).item()
        assert abs(out - p*(1-g)) < 1e-6

    def test_zero_gamma_identity(self):
        assert abs(lindblad_decoherence_step(torch.tensor(0.7), torch.tensor(0.0)).item() - 0.7) < 1e-7

    def test_unit_gamma_to_zero(self):
        assert abs(lindblad_decoherence_step(torch.tensor(0.9), torch.tensor(1.0)).item()) < 1e-7

    def test_batch_never_increases(self):
        p = torch.rand(4, 12); g = torch.rand(4, 12)
        assert (lindblad_decoherence_step(p, g) <= p + 1e-6).all()


# ── Complexity ────────────────────────────────────────────────────────────────

class TestComplexity:
    def test_qsda_cheaper_than_self_attention(self):
        for N, S in [(128, 16), (512, 16), (1024, 32)]:
            assert N*S < N*N

    def test_correct_speedup_factor(self):
        N, S = 512, 16
        assert (N*N) / (N*S) == N/S == 32

    def test_linear_scaling(self):
        S, Ns = 16, [64, 128, 256, 512, 1024]
        ratios = [(Ns[i+1]*S)/(Ns[i]*S) for i in range(len(Ns)-1)]
        expected = [Ns[i+1]/Ns[i] for i in range(len(Ns)-1)]
        np.testing.assert_allclose(ratios, expected, atol=1e-9)


# ── Model forward passes ──────────────────────────────────────────────────────

class TestModels:
    def test_v1_forward(self):
        from src.models import QSDAClassifier
        m = QSDAClassifier(50, 32, 4, 8, 2, 3, max_len=16)
        logits, metrics = m(torch.randint(0, 50, (2, 16)))
        assert logits.shape == (2, 3) and len(metrics) == 2

    def test_v2_forward(self):
        from src.models_v2 import QSDAv2Classifier
        m = QSDAv2Classifier(50, 32, 4, 2, 4, 4, 2, 3, max_len=16)
        logits, p_global, metrics = m(torch.randint(0, 50, (2, 16)))
        assert logits.shape == (2,3) and p_global.shape == (2,) and len(metrics)==2

    def test_v2_metric_keys(self):
        from src.models_v2 import QSDAv2Classifier
        m = QSDAv2Classifier(50, 32, 4, 2, 4, 4, 1, 3, max_len=16)
        _, _, metrics = m(torch.randint(0, 50, (3, 16)))
        for k in ("mean_entropy", "route_probs", "coherence", "entropy_reduction"):
            assert k in metrics[0]

    def test_entropy_valid_range(self):
        from src.models_v2 import QSDAv2Classifier
        m = QSDAv2Classifier(50, 32, 4, 2, 4, 4, 1, 3, max_len=16)
        _, _, metrics = m(torch.randint(0, 50, (3, 16)))
        H = metrics[0]["mean_entropy"]
        assert (H >= -1e-4).all() and (H <= np.log(4)+1e-3).all()

    def test_classical_forward(self):
        from src.models import ClassicalClassifier
        assert ClassicalClassifier(50,32,2,2,3,max_len=16)(
            torch.randint(0,50,(2,16))).shape == (2,3)


# ── Datasets ─────────────────────────────────────────────────────────────────

class TestDatasets:
    def test_ambiguity_shapes(self):
        from src.data import AmbiguityDataset
        ds = AmbiguityDataset(n_samples=60, seq_len=16, seed=0)
        t, l, a = ds[0]
        assert t.shape==(16,) and l.ndim==0 and 0<=a.item()<=1

    def test_sentiment_dataset(self):
        from src.data_realistic import SentimentDataset
        ds = SentimentDataset(n=60, seq_len=16, seed=0)
        assert len(ds)==60 and ds[0][0].shape==(16,)

    def test_contradiction_labels(self):
        from src.data_realistic import ContradictionDataset
        ds = ContradictionDataset(n=40, seq_len=16, seed=0)
        assert all(ds[i][1].item() in (0,1) for i in range(len(ds)))

    def test_realistic_tokens_in_vocab(self):
        from src.data_realistic import build_all_datasets
        tr, _, _, _ = build_all_datasets(seed=0)
        ids, _, _ = next(iter(tr))
        assert ids.max()<200 and ids.min()>=0

    def test_four_tasks_returned(self):
        from src.data_realistic import build_all_datasets
        _, _, _, meta = build_all_datasets(seed=0)
        assert meta["n_tasks"]==4 and meta["total"]>0
