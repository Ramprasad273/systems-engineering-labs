"""Unit tests for training utilities: LR scheduler and optimizer configuration.

These tests verify the mathematical correctness of the learning rate schedule
and the optimizer parameter grouping — two components that directly control
training stability and convergence.

Test inventory
--------------
LR Schedule (get_lr)
    test_warmup_starts_near_zero          — step 0 LR ≈ max_lr / warmup_steps
    test_warmup_reaches_max_lr            — at warmup_steps, LR == max_lr
    test_warmup_is_monotonically_increasing — LR rises each step during warmup
    test_post_warmup_is_monotonically_decreasing — LR falls each step after peak
    test_lr_never_below_min_lr            — floor constraint respected at all steps
    test_lr_equals_min_at_max_steps       — LR bottoms at min_lr when step > max_steps

Optimizer Configuration (configure_optimizers)
    test_2d_params_receive_weight_decay   — linear projection weights get λ
    test_1d_params_bypass_weight_decay    — biases / RMSNorm scales get λ=0
    test_all_trainable_params_covered     — no parameter is lost or duplicated
"""

import math
import pytest
import torch

from train import get_lr, configure_optimizers
from src.models.gpt2 import GPT2Config, GPT2Model


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

WARMUP  = 100
MAX_ST  = 1000
MAX_LR  = 6e-4
MIN_LR  = 6e-5


class TestLRSchedule:

    def test_warmup_starts_near_zero(self):
        lr = get_lr(0, WARMUP, MAX_ST, MAX_LR, MIN_LR)
        expected = MAX_LR * 1 / WARMUP
        assert math.isclose(lr, expected, rel_tol=1e-6)

    def test_warmup_reaches_max_lr(self):
        """At exactly ``warmup_steps``, the LR must equal ``max_lr``."""
        lr = get_lr(WARMUP, WARMUP, MAX_ST, MAX_LR, MIN_LR)
        assert math.isclose(lr, MAX_LR, rel_tol=1e-6)

    def test_warmup_is_monotonically_increasing(self):
        lrs = [get_lr(s, WARMUP, MAX_ST, MAX_LR, MIN_LR) for s in range(WARMUP)]
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], (
                f"LR decreased during warmup at step {i}: {lrs[i-1]:.6e} → {lrs[i]:.6e}"
            )

    def test_cosine_decay_is_monotonically_decreasing(self):
        steps = range(WARMUP, MAX_ST + 1)
        lrs = [get_lr(s, WARMUP, MAX_ST, MAX_LR, MIN_LR) for s in steps]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], (
                f"LR increased during cosine decay at step {WARMUP + i}: "
                f"{lrs[i-1]:.6e} → {lrs[i]:.6e}"
            )

    def test_lr_never_below_min_lr_post_warmup(self):
        """The min_lr floor constraint applies ONLY after the warmup phase.

        During linear warmup (steps 0 → warmup_steps) the LR intentionally
        rises from near-zero to max_lr, so it legitimately passes through
        values below min_lr. The floor is a cosine-decay constraint, not a
        global constraint across all training phases."""
        # Only check post-warmup steps where the floor must hold
        steps_post_warmup = range(WARMUP, MAX_ST + 500)
        for s in steps_post_warmup:
            lr = get_lr(s, WARMUP, MAX_ST, MAX_LR, MIN_LR)
            assert lr >= MIN_LR - 1e-9, (
                f"LR {lr:.6e} dropped below min_lr {MIN_LR:.6e} at post-warmup step {s}"
            )

    def test_warmup_phase_lr_is_below_min_lr(self):
        """During warmup, LR starts near zero and climbs to max_lr — values
        below min_lr are expected and correct during this phase."""
        lr_step_0 = get_lr(0, WARMUP, MAX_ST, MAX_LR, MIN_LR)
        assert lr_step_0 < MIN_LR, (
            f"Expected LR at step 0 ({lr_step_0:.6e}) to be below min_lr ({MIN_LR:.6e}) "
            f"during warmup. This is correct scheduler behaviour."
        )

    def test_lr_equals_min_at_beyond_max_steps(self):
        lr = get_lr(MAX_ST + 999, WARMUP, MAX_ST, MAX_LR, MIN_LR)
        assert math.isclose(lr, MIN_LR, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Optimizer Configuration
# ---------------------------------------------------------------------------

class TestOptimizerConfig:

    @pytest.fixture(scope="class")
    def model_and_groups(self, tiny_config):
        model = GPT2Model(tiny_config)
        optimizer = configure_optimizers(
            model, weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95)
        )
        return model, optimizer.param_groups

    def test_2d_params_receive_weight_decay(self, model_and_groups):
        _, groups = model_and_groups
        decay_group = groups[0]
        assert math.isclose(decay_group["weight_decay"], 0.1)
        assert len(decay_group["params"]) > 0
        # All tensors in this group must be ≥ 2D
        for p in decay_group["params"]:
            assert p.dim() >= 2, f"1D tensor found in decay group: shape {p.shape}"

    def test_1d_params_bypass_weight_decay(self, model_and_groups):
        _, groups = model_and_groups
        nodecay_group = groups[1]
        assert math.isclose(nodecay_group["weight_decay"], 0.0)
        assert len(nodecay_group["params"]) > 0
        for p in nodecay_group["params"]:
            assert p.dim() < 2, f"2D tensor found in nodecay group: shape {p.shape}"

    def test_all_trainable_params_covered(self, model_and_groups):
        """Every trainable parameter must appear in exactly one optimizer group.
        No parameter should be silently excluded or duplicated."""
        model, groups = model_and_groups
        all_ids_in_groups = set()
        for group in groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in all_ids_in_groups, "Duplicate parameter in optimizer groups"
                all_ids_in_groups.add(pid)

        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert trainable_ids == all_ids_in_groups
