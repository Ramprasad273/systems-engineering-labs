"""Unit tests for the GPT-2 model architecture.

Covers every distinct component in isolation so that regressions in any single
sub-module produce a clear, targeted failure message.

Test inventory
--------------
RMSNorm
    test_rmsnorm_output_shape          — output shape identical to input
    test_rmsnorm_unit_rms              — activations normalized to unit RMS
    test_rmsnorm_learnable_scale       — weight parameter changes output

RotaryEmbedding (RoPE)
    test_rope_output_shape             — shape preserved after rotation
    test_rope_position_zero_is_identity— position 0 keeps vector unchanged
    test_rope_different_positions_differ — rotations are position-dependent

SwiGLU
    test_swiglu_output_shape           — shape preserved through gating

CausalSelfAttention
    test_causal_masking_no_future_leak — mutation at T=t doesn't affect t'<t
    test_attention_output_shape        — output shape matches input

GPT2Model
    test_weight_tying                  — embedding weight shared with lm_head
    test_forward_no_targets_returns_none_loss — loss is None without targets
    test_forward_with_targets_loss_positive   — loss > 0 when targets provided
    test_forward_logits_shape          — logits shape (B, T, V)
    test_config_from_dict_roundtrip    — YAML dict → config preserves values
    test_parameter_count               — parameter count is deterministic
"""

import math
import torch
import pytest

from src.models.gpt2 import (
    GPT2Config,
    GPT2Model,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    CausalSelfAttention,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:

    def test_rmsnorm_output_shape(self, tiny_config):
        norm = RMSNorm(tiny_config.n_embd, eps=tiny_config.layer_norm_epsilon)
        x = torch.randn(4, 10, tiny_config.n_embd)
        assert norm(x).shape == x.shape

    def test_rmsnorm_unit_rms(self, tiny_config):
        """After normalization the RMS of each vector should be ≈ 1.0.

        The learnable weight is initialized to all-ones, so the output should
        have unit root-mean-square along the last dimension.
        """
        norm = RMSNorm(tiny_config.n_embd, eps=tiny_config.layer_norm_epsilon)
        x = torch.randn(2, 8, tiny_config.n_embd)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        # All vectors should have RMS ≈ 1 (tolerance: 1e-3)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_rmsnorm_learnable_scale(self, tiny_config):
        """Scaling the weight parameter by 2 should double the output values."""
        norm = RMSNorm(tiny_config.n_embd, eps=tiny_config.layer_norm_epsilon)
        x = torch.randn(1, 4, tiny_config.n_embd)

        with torch.no_grad():
            out_default = norm(x).clone()
            norm.weight.fill_(2.0)
            out_scaled = norm(x)

        assert torch.allclose(out_scaled, out_default * 2, atol=1e-5)


# ---------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# ---------------------------------------------------------------------------

class TestRotaryEmbedding:

    def test_rope_output_shape(self, tiny_config):
        head_dim = tiny_config.n_embd // tiny_config.n_head
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=tiny_config.block_size)
        # Shape: (batch, seq_len, n_head, head_dim)
        q = torch.randn(2, 10, tiny_config.n_head, head_dim)
        assert rope(q, seq_len=10).shape == q.shape

    def test_rope_position_zero_is_identity(self, tiny_config):
        """At sequence position 0 the rotation angle is 0, so RoPE must be
        the identity transformation: x_rot[0] == x[0]."""
        head_dim = tiny_config.n_embd // tiny_config.n_head
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=tiny_config.block_size)
        q = torch.randn(2, 10, tiny_config.n_head, head_dim)
        q_rot = rope(q, seq_len=10)
        assert torch.allclose(q_rot[:, 0, :, :], q[:, 0, :, :], atol=1e-5)

    def test_rope_different_positions_differ(self, tiny_config):
        """Non-zero positions must produce genuinely different rotations than
        position 0, confirming that positional information is injected."""
        head_dim = tiny_config.n_embd // tiny_config.n_head
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=tiny_config.block_size)
        q = torch.ones(1, 5, tiny_config.n_head, head_dim)
        q_rot = rope(q, seq_len=5)
        # Position 1 must differ from position 0
        assert not torch.allclose(q_rot[:, 0], q_rot[:, 1], atol=1e-4)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class TestSwiGLU:

    def test_swiglu_output_shape(self, tiny_config):
        ffn = SwiGLU(d_model=tiny_config.n_embd, d_ff=tiny_config.d_ff)
        x = torch.randn(3, 10, tiny_config.n_embd)
        assert ffn(x).shape == x.shape


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:

    def test_causal_masking_no_future_leak(self, tiny_config):
        """Autoregressive invariant: mutating token at position T must not
        alter the output at any earlier position T' < T.

        This is the most critical correctness property of a causal LM.
        """
        attn = CausalSelfAttention(tiny_config)
        attn.eval()

        seq_len = tiny_config.block_size
        x = torch.randn(1, seq_len, tiny_config.n_embd)

        with torch.no_grad():
            out_base = attn(x)

            # Inject a large perturbation at the last token only
            x_perturbed = x.clone()
            x_perturbed[0, -1, :] += 100.0
            out_perturbed = attn(x_perturbed)

        # All positions except the last must be completely unaffected
        assert torch.allclose(
            out_base[0, :-1, :],
            out_perturbed[0, :-1, :],
            atol=1e-4,
        ), "Causal masking violated: future token mutation leaked into past positions."

    def test_attention_output_shape(self, tiny_config):
        attn = CausalSelfAttention(tiny_config)
        x = torch.randn(2, tiny_config.block_size, tiny_config.n_embd)
        assert attn(x).shape == x.shape


# ---------------------------------------------------------------------------
# GPT2Model
# ---------------------------------------------------------------------------

class TestGPT2Model:

    def test_weight_tying(self, tiny_model):
        """Input embedding weight must share the exact same tensor as the
        language model head weight — this halves the parameter count and
        enforces consistent token representations across the model."""
        assert tiny_model.token_embeddings.weight is tiny_model.lm_head.weight

    def test_forward_no_targets_returns_none_loss(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size))
        with torch.no_grad():
            logits, loss = tiny_model(x)
        assert loss is None

    def test_forward_with_targets_loss_positive(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size))
        targets = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size))
        with torch.no_grad():
            _, loss = tiny_model(x, targets)
        assert loss is not None
        assert loss.item() > 0.0

    def test_forward_logits_shape(self, tiny_model, tiny_config):
        batch, seq = 3, tiny_config.block_size
        x = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            logits, _ = tiny_model(x)
        assert logits.shape == (batch, seq, tiny_config.vocab_size)

    def test_config_from_dict_roundtrip(self):
        """GPT2Config.from_dict() must faithfully reconstruct all fields from
        a YAML-style nested dictionary without data loss or silent coercion."""
        cfg_dict = {
            "tokenizer": {"vocab_size": 1234},
            "dataset":   {"seq_len": 128},
            "model": {
                "n_embd": 256,
                "n_layer": 4,
                "n_head": 8,
                "d_ff": 512,
                "layer_norm_epsilon": 1e-6,
            },
        }
        cfg = GPT2Config.from_dict(cfg_dict)
        assert cfg.vocab_size == 1234
        assert cfg.block_size == 128
        assert cfg.n_embd == 256
        assert cfg.n_layer == 4
        assert cfg.n_head == 8
        assert cfg.d_ff == 512
        assert math.isclose(cfg.layer_norm_epsilon, 1e-6)

    def test_parameter_count_is_deterministic(self, tiny_config):
        """Two separately instantiated models with identical configs must have
        the same total parameter count."""
        m1 = GPT2Model(tiny_config)
        m2 = GPT2Model(tiny_config)
        count = lambda m: sum(p.numel() for p in m.parameters())
        assert count(m1) == count(m2)
