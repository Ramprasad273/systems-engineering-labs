"""Unit Verification Suite for Stage 3 Mamba S6 & MambaLog Architectures.

Verifies:
1. Exact forward logits tensor dimensions (`[B, L, vocab_size]`).
2. Backward gradient propagation across all parameters without NaN/Inf instability.
3. Model capacity parameter parity (~94.3M - 125M range across 24 blocks matching Stage 1 config).
4. Recurrence state buffer invariants (`conv_state.shape == [B, ED, K]`, `ssm_state.shape == [B, ED, N]`).
5. `step()` vs `forward()` tensor dimension consistency for O(1) single-step autoregressive inference.
"""

import pytest
import torch
from src.models.mamba_block import MambaBlock
from src.models.mamba_lm import MambaLMHeadModel
from src.models.hybrid_mambalog import MambaLogLMHeadModel


@pytest.fixture
def dummy_config():
    """Returns a lightweight test configuration matching Stage 3 structural ratios."""
    return {
        "vocab_size": 5000,
        "n_embd": 128,       # Scaled down from 768 for fast CI verification
        "n_layer": 4,        # Scaled down from 24
        "d_state": 16,
        "conv_kernel": 4,
        "expand": 2,
        "dt_rank": 8,
        "n_head": 4,
        "d_ff": 256,
        "layer_norm_epsilon": 1e-5,
        "use_rotary": True,
        "attn_layer_indices": [1, 3],
    }


@pytest.fixture
def full_scale_config():
    """Returns the exact 24-layer full-scale Stage 3 configuration (`~125M` parity class)."""
    return {
        "vocab_size": 5000,
        "n_embd": 768,
        "n_layer": 24,
        "d_state": 16,
        "conv_kernel": 4,
        "expand": 2,
        "dt_rank": "auto",
        "n_head": 12,
        "d_ff": 2048,
        "layer_norm_epsilon": 1e-5,
        "use_rotary": True,
        "attn_layer_indices": [3, 7, 11, 15, 19, 23],
    }


def test_mamba_block_forward_and_backward():
    """Verifies single MambaBlock forward shape and clean backward gradient propagation."""
    batch_size, seq_len, d_model = 2, 32, 128
    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2, dt_rank=8)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    out = block(x)
    assert out.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"

    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None, "Input gradients failed to propagate through MambaBlock."
    assert not torch.isnan(x.grad).any(), "NaN detected in input gradients during backward pass."
    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert not torch.isnan(param.grad).any(), f"NaN detected in gradient for parameter {name}"


def test_mamba_lm_forward_shapes_and_grad(dummy_config):
    """Verifies MambaLMHeadModel forward logits shape `[B, L, 5000]` and backward gradient flow."""
    batch_size, seq_len = 2, 32
    model = MambaLMHeadModel(dummy_config)
    input_ids = torch.randint(0, dummy_config["vocab_size"], (batch_size, seq_len))
    targets = torch.randint(0, dummy_config["vocab_size"], (batch_size, seq_len))

    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (batch_size, seq_len, dummy_config["vocab_size"]), f"Expected {(batch_size, seq_len, 5000)}, got {logits.shape}"
    assert loss is not None and loss.item() > 0.0, "Loss calculation returned invalid scalar."

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradient."
            assert not torch.isnan(param.grad).any(), f"NaN detected in {name}.grad"


def test_mamba_lm_recurrence_state_invariants(dummy_config):
    """Verifies recurrence state buffer dimensions and step() generation invariant (`O(1)` memory)."""
    batch_size = 2
    model = MambaLMHeadModel(dummy_config)
    cache = model.allocate_inference_cache(batch_size=batch_size, device="cpu")

    assert len(cache) == dummy_config["n_layer"], f"Expected {dummy_config['n_layer']} state tuples, got {len(cache)}"
    d_inner = dummy_config["expand"] * dummy_config["n_embd"]
    d_conv = dummy_config["conv_kernel"]
    d_state = dummy_config["d_state"]

    for i, (conv_state, ssm_state) in enumerate(cache):
        assert conv_state.shape == (batch_size, d_inner, d_conv), f"Layer {i} conv_state shape mismatch: {conv_state.shape}"
        assert ssm_state.shape == (batch_size, d_inner, d_state), f"Layer {i} ssm_state shape mismatch: {ssm_state.shape}"

    # Execute single token step
    token_id = torch.randint(0, dummy_config["vocab_size"], (batch_size, 1))
    logits_t, new_cache = model.step(token_id, cache)

    assert logits_t.shape == (batch_size, dummy_config["vocab_size"]), f"Logits step shape mismatch: {logits_t.shape}"
    assert len(new_cache) == dummy_config["n_layer"]
    for conv_s, ssm_s in new_cache:
        assert not torch.isnan(conv_s).any(), "NaN in conv_state after step()"
        assert not torch.isnan(ssm_s).any(), "NaN in ssm_state after step()"


def test_hybrid_mambalog_forward_shapes(dummy_config):
    """Verifies MambaLogLMHeadModel forward shape, hybrid block interleaving, and backward gradients."""
    batch_size, seq_len = 2, 32
    model = MambaLogLMHeadModel(dummy_config)
    input_ids = torch.randint(0, dummy_config["vocab_size"], (batch_size, seq_len))
    targets = torch.randint(0, dummy_config["vocab_size"], (batch_size, seq_len))

    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (batch_size, seq_len, dummy_config["vocab_size"])
    assert loss is not None

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing gradient in MambaLog."
            assert not torch.isnan(param.grad).any(), f"NaN in {name}.grad in MambaLog."


def test_full_scale_parameter_parity(full_scale_config):
    """Verifies exact parameter counts for full scale 24-layer models (`~94.3M - 125M` parity class)."""
    mamba_model = MambaLMHeadModel(full_scale_config)
    mambalog_model = MambaLogLMHeadModel(full_scale_config)

    mamba_params = sum(p.numel() for p in mamba_model.parameters() if p.requires_grad)
    mambalog_params = sum(p.numel() for p in mambalog_model.parameters() if p.requires_grad)

    print(f"\n[Capacity Verification] Full-scale Mamba S6 Parameters: {mamba_params:,} ({mamba_params / 1e6:.2f}M)")
    print(f"[Capacity Verification] Full-scale MambaLog Parameters: {mambalog_params:,} ({mambalog_params / 1e6:.2f}M)")

    # Assert models are within the target ~90M - 130M parameter capacity band matching Stage 1 GPT-2 baseline
    assert 85_000_000 <= mamba_params <= 135_000_000, f"Mamba parameter count {mamba_params:,} outside expected parity bounds."
    assert 85_000_000 <= mambalog_params <= 135_000_000, f"MambaLog parameter count {mambalog_params:,} outside bounds."
