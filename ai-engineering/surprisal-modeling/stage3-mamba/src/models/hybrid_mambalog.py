"""Hybrid MambaLog Language Model (`3:1` Interleaved Mamba:Attention).

Pedagogical engineering following Karpathy guidelines:
- Interleaves 18 Mamba blocks with 6 Causal Self-Attention (RoPE + SwiGLU) blocks (`indices: [3, 7, 11, 15, 19, 23]`).
- Evaluates syntax efficiency of S6 vs long-context template stability of causal attention across 8,192-token sessions.
- Explicit shape annotations and docstrings explaining WHY hybrid interleaving works.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mamba_block import RMSNorm, ResidualBlock as MambaResidualBlock


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) relative coordinate encoder matching Stage 1 parity."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        end_pos = start_pos + x.shape[1]
        if end_pos > self.cos_cached.shape[0]:
            self._update_cache(end_pos)
        cos = self.cos_cached[start_pos:end_pos, :].unsqueeze(0).unsqueeze(2).type_as(x)
        sin = self.sin_cached[start_pos:end_pos, :].unsqueeze(0).unsqueeze(2).type_as(x)
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """SwiGLU Gated Activation Feed-Forward Network matching Stage 1 parity."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism with RoPE."""

    def __init__(self, n_embd: int, n_head: int, block_size: int = 8192, use_rotary: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, "Hidden dimension must be divisible by head count."
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rotary = use_rotary

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        if self.use_rotary:
            self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        if self.use_rotary:
            q = self.rope(q, seq_len, start_pos=0)
            k = self.rope(k, seq_len, start_pos=0)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_out)


class AttentionResidualBlock(nn.Module):
    """Pre-Layer Normalization Causal Attention + SwiGLU FFN block."""

    def __init__(self, config: dict):
        super().__init__()
        n_embd = config.get("n_embd", 768)
        n_head = config.get("n_head", 12)
        d_ff = config.get("d_ff", 2048)
        eps = config.get("layer_norm_epsilon", 1e-5)
        block_size = config.get("block_size", 8192)

        self.attn_norm = RMSNorm(n_embd, eps=eps)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            use_rotary=config.get("use_rotary", True),
        )

        self.ffn_norm = RMSNorm(n_embd, eps=eps)
        self.ffn = SwiGLU(n_embd, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, d_model]
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_kv_len: int = 512,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step recurrent attention with sliding-window bounded KV cache.

        KV cache is capped at `max_kv_len` (default=512, matching training seq_len)
        using a sliding window to maintain O(1) VRAM cost independent of step count.
        """
        normed = self.attn_norm(x_t)  # [batch_size, n_embd]
        batch_size = normed.shape[0]
        q = self.attn.q_proj(normed).view(batch_size, 1, self.attn.n_head, self.attn.head_dim)
        k = self.attn.k_proj(normed).view(batch_size, 1, self.attn.n_head, self.attn.head_dim)
        v = self.attn.v_proj(normed).view(batch_size, 1, self.attn.n_head, self.attn.head_dim)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]
        if self.attn.use_rotary:
            q = self.attn.rope(q, 1, start_pos=past_len)
            k = self.attn.rope(k, 1, start_pos=past_len)

        q = q.transpose(1, 2)  # [batch_size, n_head, 1, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # Sliding-window cap: keep only the most recent `max_kv_len` tokens
        # This bounds VRAM to O(max_kv_len) regardless of decoding steps.
        if k.shape[2] > max_kv_len:
            k = k[:, :, -max_kv_len:, :]
            v = v[:, :, -max_kv_len:, :]
        new_kv = (k, v)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.attn.n_head * self.attn.head_dim)
        attn_proj = self.attn.out_proj(attn_out)
        x_mid = x_t + attn_proj
        return x_mid + self.ffn(self.ffn_norm(x_mid)), new_kv


class MambaLogLMHeadModel(nn.Module):
    """Hybrid MambaLog Architecture interleaving S6 blocks with Causal Attention blocks (`18:6` or `21:3`).

    Attributes:
        token_embeddings (nn.Embedding): Token lookup table `[vocab_size, n_embd]`.
        layers (nn.ModuleList): Stack of 24 interleaved Mamba and Attention blocks.
        norm_f (RMSNorm): Final root mean square layer normalization.
        lm_head (nn.Linear): Weight-tied linear classification projection `[n_embd, vocab_size]`.
    """

    def __init__(self, config: dict | object):
        super().__init__()
        if not isinstance(config, dict):
            cfg_dict = {
                "vocab_size": getattr(config, "vocab_size", 5000),
                "n_embd": getattr(config, "n_embd", 768),
                "n_layer": getattr(config, "n_layer", 24),
                "d_state": getattr(config, "d_state", 16),
                "conv_kernel": getattr(config, "conv_kernel", 4),
                "expand": getattr(config, "expand", 2),
                "dt_rank": getattr(config, "dt_rank", "auto"),
                "n_head": getattr(config, "n_head", 12),
                "d_ff": getattr(config, "d_ff", 2048),
                "layer_norm_epsilon": getattr(config, "layer_norm_epsilon", 1e-5),
                "use_rotary": getattr(config, "use_rotary", True),
                "attn_layer_indices": getattr(config, "attn_layer_indices", [3, 7, 11, 15, 19, 23]),
            }
        else:
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
                    "n_head": model_cfg.get("n_head", 12),
                    "d_ff": model_cfg.get("d_ff", 2048),
                    "layer_norm_epsilon": float(model_cfg.get("layer_norm_epsilon", 1e-5)),
                    "use_rotary": model_cfg.get("use_rotary", True),
                    "attn_layer_indices": model_cfg.get("attn_layer_indices", [3, 7, 11, 15, 19, 23]),
                }
            else:
                cfg_dict = config

        self.config = cfg_dict
        vocab_size = cfg_dict.get("vocab_size", 5000)
        n_embd = cfg_dict.get("n_embd", 768)
        n_layer = cfg_dict.get("n_layer", 24)
        eps = float(cfg_dict.get("layer_norm_epsilon", 1e-5))
        attn_indices = set(cfg_dict.get("attn_layer_indices", [3, 7, 11, 15, 19, 23]))

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        
        layers = []
        for i in range(n_layer):
            if i in attn_indices:
                layers.append(AttentionResidualBlock(cfg_dict))
            else:
                layers.append(MambaResidualBlock(cfg_dict))
        self.layers = nn.ModuleList(layers)

        self.norm_f = RMSNorm(n_embd, eps=eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
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
    ) -> list:
        """Allocates zeroed recurrent and KV state buffers for hybrid blocks."""
        cache = []
        d_inner = int(self.config["expand"] * self.config["n_embd"])
        d_conv = self.config["conv_kernel"]
        d_state = self.config["d_state"]

        for layer in self.layers:
            if isinstance(layer, AttentionResidualBlock):
                cache.append(None)
            else:
                conv_state = torch.zeros(batch_size, d_inner, d_conv, device=device, dtype=dtype)
                ssm_state = torch.zeros(batch_size, d_inner, d_state, device=device, dtype=dtype)
                cache.append((conv_state, ssm_state))
        return cache

    def step(
        self,
        token_id: torch.Tensor,
        state_cache: list,
    ) -> tuple[torch.Tensor, list]:
        """Executes single-step recurrent token generation (`O(1)` Mamba + KV Attention)."""
        if token_id.dim() == 2:
            token_id = token_id.squeeze(1)

        x_t = self.token_embeddings(token_id)  # [batch_size, n_embd]
        new_cache = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, AttentionResidualBlock):
                kv = state_cache[i]
                if kv is not None and isinstance(kv, tuple) and len(kv) == 2 and kv[0].dim() == 4 and kv[0].shape[1] == layer.attn.n_head:
                    pass
                else:
                    kv = None
                x_t, new_kv = layer.step(x_t, kv)
                new_cache.append(new_kv)
            else:
                conv_state, ssm_state = state_cache[i]
                x_t, new_conv, new_ssm = layer.step(x_t, conv_state, ssm_state)
                new_cache.append((new_conv, new_ssm))

        x_t = self.norm_f(x_t)
        logits_t = self.lm_head(x_t)  # [batch_size, vocab_size]
        return logits_t, new_cache
