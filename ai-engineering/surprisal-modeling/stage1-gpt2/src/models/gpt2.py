"""Custom Lightweight GPT-2 Autoregressive Transformer Architecture.

Pedagogical explanations of architectural departures (RoPE vs absolute pos embeddings,
SwiGLU vs GELU, RMSNorm vs LayerNorm), explicit tensor shape annotations [batch_size, seq_len, dim],
and academic weight initialization.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    """Configuration hyperparameters for the custom GPT-2 architecture.

    Attributes:
        vocab_size (int): Vocabulary limit matching trained BPE tokenizer capacity.
        n_embd (int): Hidden dimension width ($d_{\text{model}}$).
        n_layer (int): Number of stacked transformer blocks.
        n_head (int): Number of attention heads across query/key/value projections.
        block_size (int): Maximum sequence length capacity ($T$).
        d_ff (int): Inner intermediate projection dimension for FFN blocks.
        layer_norm_epsilon (float): Numerical stability variance epsilon for RMSNorm.
        use_rotary (bool): Whether to use RoPE relative embeddings (True) or learned absolute embeddings (False).
        use_swiglu (bool): Whether to use SwiGLU gated activation (True) or standard GELU MLP (False).
    """
    vocab_size: int = 5000
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 512
    d_ff: int = 2048
    layer_norm_epsilon: float = 1e-5
    use_rotary: bool = True
    use_swiglu: bool = True

    def __post_init__(self):
        self.layer_norm_epsilon = float(self.layer_norm_epsilon)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPT2Config":
        model_cfg = config_dict.get("model", {})
        tokenizer_cfg = config_dict.get("tokenizer", {})
        dataset_cfg = config_dict.get("dataset", {})
        
        return cls(
            vocab_size=tokenizer_cfg.get("vocab_size", 5000),
            n_embd=model_cfg.get("n_embd", 768),
            n_layer=model_cfg.get("n_layer", 12),
            n_head=model_cfg.get("n_head", 12),
            block_size=dataset_cfg.get("seq_len", 512),
            d_ff=model_cfg.get("d_ff", 2048),
            layer_norm_epsilon=float(model_cfg.get("layer_norm_epsilon", 1e-5)),
            use_rotary=model_cfg.get("use_rotary", True),
            use_swiglu=model_cfg.get("use_swiglu", True)
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    WHY: Removing mean-centering invariance simplifies hardware execution and saves memory bandwidth
    without hurting gradient backpropagation stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) relative coordinate encoder."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max(max_seq_len, 2048))

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

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2).type_as(x)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2).type_as(x)
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """SwiGLU Gated Activation Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GeluMLP(nn.Module):
    """Standard GELU Multilayer Perceptron fallback for ablation study."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff, bias=False)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism supporting RoPE or absolute embeddings."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Hidden dimension must be divisible by head count."
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rotary = config.use_rotary
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        if self.use_rotary:
            self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=config.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        
        if self.use_rotary:
            q = self.rope(q, seq_len)
            k = self.rope(k, seq_len)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_out)


class GPT2Block(nn.Module):
    """Transformer block following Pre-Layer Normalization topology."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        if config.use_swiglu:
            self.ffn = SwiGLU(config.n_embd, config.d_ff)
        else:
            self.ffn = GeluMLP(config.n_embd, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT2Model(nn.Module):
    """Custom Autoregressive GPT-2 Small neural network."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.use_rotary:
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
            
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        x = self.token_embeddings(input_ids)
        
        if not self.config.use_rotary:
            positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            x = x + self.pos_embeddings(positions)
            
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss
