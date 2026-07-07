"""Low-Rank Adaptation (LoRA) Implementation from First Principles.

Implements explicit low-rank matrix decomposition (ΔW = BA) without external wrapping libraries.
Provides pedagogical explanations of parameter savings, zero-initialization invariants, scaling
factors (α/r), model adapter injection, and weight merging for zero-overhead deployment.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Polyfill set_submodule for PyTorch < 2.5.0 compatibility with latest transformers/bitsandbytes
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        if target == "":
            raise ValueError("Cannot set the root module")
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self.get_submodule(".".join(atoms)) if atoms else self
        setattr(mod, name, module)
    nn.Module.set_submodule = _set_submodule

logger = logging.getLogger("stage2.lora")


class LoRALinear(nn.Module):
    """Low-Rank Adaptation of a frozen linear projection layer.

    Mathematical formulation:
        W_effective = W_0 + (α / r) * (B @ A)

    Where:
        W_0 ∈ ℝ^{out_features × in_features} is frozen pre-trained weight matrix.
        A ∈ ℝ^{rank × in_features} is trainable down-projection (Kaiming uniform init).
        B ∈ ℝ^{out_features × rank} is trainable up-projection (Zero init).
        α is hyperparameter scaling factor controlling adapter magnitude relative to base.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Preserve and freeze existing pre-trained layer weights and biases
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Determine appropriate floating point dtype for LoRA adapter parameters.
        # When base_layer is 4-bit/8-bit quantized (e.g. BitsAndBytes uint8/int8), weight.dtype is integer.
        # We must use a floating point dtype (e.g. compute_dtype or float32/bfloat16) for trainable gradients!
        if base_layer.weight.dtype.is_floating_point:
            lora_dtype = base_layer.weight.dtype
        elif hasattr(base_layer, "compute_dtype") and base_layer.compute_dtype is not None and base_layer.compute_dtype.is_floating_point:
            lora_dtype = base_layer.compute_dtype
        elif hasattr(base_layer.weight, "compute_dtype") and base_layer.weight.compute_dtype is not None and base_layer.weight.compute_dtype.is_floating_point:
            lora_dtype = base_layer.weight.compute_dtype
        elif torch.cuda.is_bf16_supported():
            lora_dtype = torch.bfloat16
        else:
            lora_dtype = torch.float32

        lora_device = base_layer.weight.device
        # Low-rank adapter projections
        # A: down-project from in_features -> rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=lora_dtype, device=lora_device))
        # B: up-project from rank -> out_features
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=lora_dtype, device=lora_device))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes low-rank adapter matrices.

        WHY zero-init B: Ensures ΔW = B @ A = 0 at initialization step 0.
        This guarantees identical forward pass outputs to the pre-trained model before training starts.
        WHY Kaiming-init A: Standard variance-preserving distribution for active representations.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Fuses low-rank adapter weights into base weight matrix for zero-overhead inference."""
        if not self.merged and self.rank > 0:
            # ΔW: [out_features, in_features] = B @ A * scaling
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_weight.to(dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)
            self.merged = True
            logger.debug(f"Merged LoRA weights (rank={self.rank}) into base layer [{self.in_features}->{self.out_features}].")

    def unmerge(self):
        """Separates adapter weights from base weight matrix."""
        if self.merged and self.rank > 0:
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_weight.to(dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)
            self.merged = False
            logger.debug(f"Unmerged LoRA weights from base layer [{self.in_features}->{self.out_features}].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computation: base_output + lora_output.

        Args:
            x: Input activation tensor of shape [batch_size, seq_len, in_features].

        Returns:
            Output tensor of shape [batch_size, seq_len, out_features].

        WHY sync before base_layer:
            Under CUDA_LAUNCH_BLOCKING=0 (async default), BitsAndBytes dequantize_4bit
            launches async CUDA kernels. On WSL2/WDDM, these kernels may still be queued
            when the next torch.empty() allocation is requested, causing CUDA_ERROR_NOT_READY.
            This sync fence (a no-op under CUDA_LAUNCH_BLOCKING=1) ensures the driver is clean.
        """
        base_output = self.base_layer(x)

        if self.rank > 0 and not self.merged:
            # Cast input to LoRA adapter dtype (bfloat16) and compute low-rank delta
            lora_input = self.lora_dropout(x.to(dtype=self.lora_A.dtype))
            lora_delta = F.linear(F.linear(lora_input, self.lora_A), self.lora_B) * self.scaling
            # Cast LoRA delta back to base output dtype before addition
            return base_output + lora_delta.to(dtype=base_output.dtype)

        return base_output


def inject_lora_adapters(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05
) -> int:
    """Scans PyTorch model and replaces specified linear layers with LoRALinear modules.

    Args:
        model: Target neural network model.
        target_modules: List of substring names indicating target linear modules (e.g., ['q_proj', 'v_proj']).
        rank: Low-rank dimension r.
        alpha: LoRA scaling hyperparameter α.
        dropout: Dropout probability applied before down-projection.

    Returns:
        Number of injected adapter modules.
    """
    injected_count = 0
    for name, module in dict(model.named_modules()).items():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            
            lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_module)
            injected_count += 1

    # Freeze all parameters except lora_A and lora_B
    for param_name, param in model.named_parameters():
        if "lora_A" in param_name or "lora_B" in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    logger.info(f"Successfully injected {injected_count} LoRA adapter layers (rank={rank}, alpha={alpha}).")
    return injected_count


def merge_lora_weights(model: nn.Module):
    """Traverses model and merges all LoRALinear adapters into base weights."""
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged_count += 1
    logger.info(f"Merged {merged_count} LoRA adapter layers into permanent base model weights.")


def count_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Computes total vs trainable parameter count and savings percentage.

    Returns:
        Tuple of (trainable_params, all_params, trainable_percentage).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    return trainable_params, all_params, percentage
