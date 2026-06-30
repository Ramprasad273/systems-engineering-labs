"""Integration test: overfitting sanity check.

Before training at scale, verify that the model can memorize a tiny fixed batch.
If it cannot, there is a bug in the forward or backward pass — not a data or hyperparameter issue.

The test uses a deliberately small model and vocabulary to ensure convergence
within 200 steps on CPU, making it suitable for CI without GPU access.

Failure modes that this test catches
--------------------------------------
- Incorrect causal masking (information leak → unusually fast loss drop or
  the loss plateau above 0.1 on synthetic random data)
- Disconnected backward graph (no gradient flow → loss never decreases)
- Weight initialization issues (NaN loss from step 0)
- Broken optimizer parameter groups (parameters not updated)
"""

import pytest
import torch

from src.models.gpt2 import GPT2Config, GPT2Model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 50
SEQ_LEN     = 32
BATCH_SIZE  = 4
NUM_STEPS   = 200
LOSS_TARGET = 0.10
LR          = 5e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_batch(device: str):
    """Returns a deterministic (inputs, targets) pair from a tiny synthetic
    token sequence.  The seed is fixed so the test is reproducible across
    different runs and hardware configurations."""
    torch.manual_seed(42)
    data    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN + 1), device=device)
    inputs  = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets


def _tiny_model(device: str) -> GPT2Model:
    cfg = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=128,
        n_layer=2,
        n_head=4,
        block_size=SEQ_LEN,
        d_ff=256,
        layer_norm_epsilon=1e-5,
    )
    return GPT2Model(cfg).to(device)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestOverfitSanity:

    def test_tiny_model_memorizes_fixed_batch(self, device, autocast_dtype):
        """The model must drive cross-entropy loss below LOSS_TARGET within
        NUM_STEPS optimization steps on a fixed synthetic batch.

        This is the canonical gradient flow sanity check: if the model cannot
        memorize 4 random sequences of length 32 over 200 steps, something
        in the forward/backward pipeline is fundamentally broken.
        """
        inputs, targets = _fixed_batch(device)
        model     = _tiny_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        model.train()

        initial_loss = None
        final_loss   = None

        for step in range(NUM_STEPS):
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if step == 0:
                initial_loss = loss_val
            final_loss = loss_val

        # Sanity: training should have started with a reasonable initial loss
        assert initial_loss is not None
        assert not torch.isnan(torch.tensor(initial_loss)), \
            "NaN loss at step 0 — check weight initialization"

        assert final_loss < LOSS_TARGET, (
            f"Overfitting sanity FAILED: final loss {final_loss:.4f} did not "
            f"reach < {LOSS_TARGET} in {NUM_STEPS} steps. "
            f"Initial loss was {initial_loss:.4f}. "
            f"This indicates a bug in the forward or backward pass."
        )

    def test_loss_decreases_from_step_1_to_step_50(self, device, autocast_dtype):
        """Loss must show a clear downward trend in the first 50 steps.

        A flat or rising loss curve from step 0 indicates that gradients are
        not flowing (e.g., disconnected computation graph or zero-LR bug).
        """
        inputs, targets = _fixed_batch(device)
        model     = _tiny_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        model.train()

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # The average loss in the last 10 steps must be lower than the first 10
        early_avg = sum(losses[:10]) / 10
        late_avg  = sum(losses[40:]) / 10
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )
