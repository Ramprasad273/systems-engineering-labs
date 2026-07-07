"""Unit tests for first-principles LoRA implementation."""

import torch
import torch.nn as nn
from src.models.lora import (
    LoRALinear,
    inject_lora_adapters,
    merge_lora_weights,
    count_trainable_parameters
)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        h = self.o_proj(h)
        return self.fc2(torch.relu(self.fc1(h)))


def test_zero_initialization_invariant():
    """Verifies that zero-init of lora_B guarantees identical forward pass at step 0."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 64)
    x = torch.randn(2, 10, 32)
    
    expected_output = base_linear(x)
    
    lora_linear = LoRALinear(base_linear, rank=8, alpha=16.0)
    actual_output = lora_linear(x)
    
    assert torch.allclose(expected_output, actual_output, atol=1e-6), "LoRA initialization must yield exact zero ΔW."


def test_parameter_savings():
    """Verifies significant parameter reduction when injecting LoRA adapters."""
    model = SimpleTransformerLayer(d_model=128)
    total_before = sum(p.numel() for p in model.parameters())
    
    injected_count = inject_lora_adapters(model, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0)
    assert injected_count == 2
    
    trainable, total_after, percentage = count_trainable_parameters(model)
    assert trainable < total_before * 0.2, f"Expected <20% trainable params, got {percentage:.2f}%"
    assert total_after > total_before


def test_merge_unmerge_invariance():
    """Verifies forward pass equivalence between adapter forward and merged weights."""
    torch.manual_seed(123)
    base_linear = nn.Linear(32, 32)
    lora_linear = LoRALinear(base_linear, rank=4, alpha=8.0)
    
    # Simulate step > 0 by perturbing weights
    with torch.no_grad():
        lora_linear.lora_B.data.normal_(0, 0.1)
        
    x = torch.randn(2, 5, 32)
    unmerged_output = lora_linear(x)
    
    lora_linear.merge()
    assert lora_linear.merged
    merged_output = lora_linear(x)
    
    assert torch.allclose(unmerged_output, merged_output, atol=1e-5), "Merged output must match unmerged adapter output."
    
    lora_linear.unmerge()
    assert not lora_linear.merged
    unmerged_again = lora_linear(x)
    assert torch.allclose(unmerged_output, unmerged_again, atol=1e-5)


if __name__ == "__main__":
    test_zero_initialization_invariant()
    test_parameter_savings()
    test_merge_unmerge_invariance()
    print("All LoRA first-principles unit tests passed successfully!")
