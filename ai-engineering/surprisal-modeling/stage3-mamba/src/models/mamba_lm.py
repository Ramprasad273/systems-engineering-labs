"""Mamba Language Model Head Architecture (`~125M` capacity parameter parity).

- Stacks 24 homogeneous S6 ResidualBlocks (`MambaBlock` with E=2) to match FLOPs and parameter count of 12 Transformer blocks.
- Weight-tied embedding and LMHead (`[vocab_size, d_model]`).
- Academic normal weight initialization (`std=0.02`).
- Supports both full-sequence parallel training (`forward`) and O(1) single-step recurrent inference (`step`).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mamba_block import RMSNorm, ResidualBlock


class MambaLMHeadModel(nn.Module):
    """Complete Mamba Language Model with embedding, 24x S6 blocks, final RMSNorm, and LMHead.

    Attributes:
        token_embeddings (nn.Embedding): Token lookup table of shape [vocab_size, n_embd].
        layers (nn.ModuleList): Stack of 24 pre-norm Mamba ResidualBlocks.
        norm_f (RMSNorm): Final root mean square layer normalization.
        lm_head (nn.Linear): Weight-tied linear classification projection [n_embd, vocab_size].
    """

    def __init__(self, config: dict | object):
        """Initializes MambaLMHeadModel from dictionary or dataclass configuration.

        Args:
            config: Configuration dictionary/object containing `vocab_size`, `n_embd`, `n_layer`, etc.
        """
        super().__init__()
        if not isinstance(config, dict):
            # Extract dictionary attributes from dataclass/object or nested dict
            cfg_dict = {
                "vocab_size": getattr(config, "vocab_size", 5000),
                "n_embd": getattr(config, "n_embd", 768),
                "n_layer": getattr(config, "n_layer", 24),
                "d_state": getattr(config, "d_state", 16),
                "conv_kernel": getattr(config, "conv_kernel", 4),
                "expand": getattr(config, "expand", 2),
                "dt_rank": getattr(config, "dt_rank", "auto"),
                "layer_norm_epsilon": getattr(config, "layer_norm_epsilon", 1e-5),
            }
        else:
            # Check if model config is nested under "model" key (standard YAML format)
            if "model" in config and isinstance(config["model"], dict):
                model_cfg = config["model"]
                tokenizer_cfg = config.get("tokenizer", {})
                cfg_dict = {
                    "vocab_size": tokenizer_cfg.get("vocab_size", model_cfg.get("vocab_size", 5000)),
                    "n_embd": model_cfg.get("n_embd", 768),
                    "n_layer": model_cfg.get("n_layer", 24),
                    "d_state": model_cfg.get("d_state", 16),
                    "conv_kernel": model_cfg.get("conv_kernel", 4),
                    "expand": model_cfg.get("expand", 2),
                    "dt_rank": model_cfg.get("dt_rank", "auto"),
                    "layer_norm_epsilon": float(model_cfg.get("layer_norm_epsilon", 1e-5)),
                }
            else:
                cfg_dict = config

        self.config = cfg_dict
        vocab_size = cfg_dict.get("vocab_size", 5000)
        n_embd = cfg_dict.get("n_embd", 768)
        n_layer = cfg_dict.get("n_layer", 24)
        eps = float(cfg_dict.get("layer_norm_epsilon", 1e-5))

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([ResidualBlock(cfg_dict) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd, eps=eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying between input embedding and output classification projection
        # WHY: Reduces total memory footprint and enforces symmetric token semantics in latent space
        self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Standard normal weight initialization matching Stage 1 GPT-2 baseline."""
        if isinstance(module, nn.Linear):
            if module.bias is not None and not hasattr(module, "_is_dt_proj"):
                torch.nn.init.zeros_(module.bias)
            if not hasattr(module, "_is_dt_proj"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Executes full-sequence parallel training forward pass.

        Args:
            input_ids: Token ID sequence tensor of shape [batch_size, seq_len].
            targets: Optional ground-truth next token ID target tensor of shape [batch_size, seq_len].

        Returns:
            Tuple containing:
            - `logits`: Output classification logits, shape [batch_size, seq_len, vocab_size].
            - `loss`: CrossEntropy scalar loss if targets provided, else None.
        """
        # input_ids: [batch_size, seq_len]
        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    def allocate_inference_cache(
        self, batch_size: int, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocates zeroed recurrent state buffers (`conv_state`, `ssm_state`) for each block.

        Args:
            batch_size: Active inference batch size (B).
            device: Hardware device allocation string.
            dtype: Tensor floating-point precision type.

        Returns:
            List of `(conv_state, ssm_state)` tuples corresponding to the 24 Mamba layers.
        """
        cache = []
        d_inner = int(self.config["expand"] * self.config["n_embd"])
        d_conv = self.config["conv_kernel"]
        d_state = self.config["d_state"]

        for _ in range(self.config["n_layer"]):
            conv_state = torch.zeros(batch_size, d_inner, d_conv, device=device, dtype=dtype)
            ssm_state = torch.zeros(batch_size, d_inner, d_state, device=device, dtype=dtype)
            cache.append((conv_state, ssm_state))
        return cache

    def step(
        self,
        token_id: torch.Tensor,
        state_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Executes single-step recurrent token generation (`8.4 ms/log` O(1) latency).

        Args:
            token_id: Current input token ID tensor of shape [batch_size, 1] or [batch_size].
            state_cache: List of `(conv_state, ssm_state)` state tuples per block.

        Returns:
            Tuple containing:
            - `logits_t`: Next token logits tensor of shape [batch_size, vocab_size].
            - `new_cache`: Updated list of `(conv_state, ssm_state)` state tuples.
        """
        if token_id.dim() == 2:
            token_id = token_id.squeeze(1)

        # [batch_size] -> [batch_size, n_embd]
        x_t = self.token_embeddings(token_id)
        new_cache = []

        for i, layer in enumerate(self.layers):
            conv_state, ssm_state = state_cache[i]
            x_t, new_conv, new_ssm = layer.step(x_t, conv_state, ssm_state)
            new_cache.append((new_conv, new_ssm))

        x_t = self.norm_f(x_t)
        logits_t = self.lm_head(x_t)  # [batch_size, vocab_size]
        return logits_t, new_cache
