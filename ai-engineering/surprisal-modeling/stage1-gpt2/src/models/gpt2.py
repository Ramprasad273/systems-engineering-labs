"""Custom Lightweight GPT-2 Autoregressive Transformer Architecture.

This module implements a modern, highly optimized variant of the GPT-2 Small transformer
tailored for sequence anomaly modeling. Key architectural departures from traditional GPT-2
include: Root Mean Square Layer Normalization (RMSNorm) for gradient stability, Rotary
Position Embeddings (RoPE) for relative sequence position encoding without learned positional
parameters, SwiGLU gated activation networks in the feed-forward blocks, and Pre-LN block layout.
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
        d_ff (int): Inner intermediate projection dimension for SwiGLU FFN blocks.
        layer_norm_epsilon (float): Numerical stability variance epsilon for RMSNorm.
    """
    vocab_size: int = 5000
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 512
    d_ff: int = 2048
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        self.layer_norm_epsilon = float(self.layer_norm_epsilon)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPT2Config":
        """Instantiates a GPT2Config object from a deserialized YAML configuration dictionary.

        Args:
            config_dict: Raw configuration mapping containing `model`, `tokenizer`, and `dataset` sections.

        Returns:
            Populated GPT2Config dataclass instance.
        """
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
            layer_norm_epsilon=float(model_cfg.get("layer_norm_epsilon", 1e-5))
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm simplifies traditional Layer Normalization by removing mean-centering invariance
    and normalizing activations solely by their reciprocal root mean square activation norm.
    This reduces memory bandwidth overhead and stabilizes deep transformer gradient backpropagation.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initializes scale parameters and numerical stability bounds.

        Args:
            dim: Hidden dimension size to normalize across.
            eps: Small numerical constant added to variance before square root.
        """
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes unscaled root mean square normalization over the final axis."""
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm in float32 precision to mitigate underflow in bfloat16 mixed-precision."""
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) relative coordinate encoder.

    RoPE injects relative positional information directly into attention dot-products by multiplying
    query and key head representations by complex orthogonal rotation matrices in 2D vector subspaces.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        """Pre-computes rotary frequency basis vectors.

        Args:
            dim: Individual attention head dimension ($d_k = d_{\text{model}} / n_{\text{head}}$).
            max_seq_len: Baseline context bound. Pre-computes up to at least 2048 for memory sweeps.
            theta: Base geometric progression constant for frequency decay.
        """
        super().__init__()
        self.dim = dim
        
        # Calculate inverse frequency geometric progression: \theta^{-2i/d}
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute rotation cos/sin lookup tables up to N=2048 to support VRAM footprint sweeps
        self._update_cache(max(max_seq_len, 2048))

    def _update_cache(self, seq_len: int):
        """Generates cached cosine and sine tensor buffers up to sequence length N."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate frequencies across even/odd pairs to match tensor head dimensions
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Negates and swaps orthogonal 2D vector subspace halves: [-x2, x1]."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies rotary coordinate transformations to query or key tensors.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_head, head_dim).
            seq_len: Active sequence length T.

        Returns:
            Positionally rotated tensor of identical dimensions.
        """
        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)
            
        # Extract active sequence bounds and broadcast across batch and head axes
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2).type_as(x)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2).type_as(x)
        
        # Euler rotation identity transformation: (x * cos) + (rotate_half(x) * sin)
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """SwiGLU Gated Activation Feed-Forward Network.

    SwiGLU enhances expressivity over standard ReLU/GELU multilayer perceptrons by combining
    an activation gate projected via SiLU (Swish) with a linear projection branch.
    """

    def __init__(self, d_model: int, d_ff: int):
        """Initializes unshared gating and projection linear transformations.

        Args:
            d_model: Input hidden dimension ($768$).
            d_ff: Expanded intermediate SwiGLU dimension ($2048$).
        """
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes element-wise gated activation projection: W_down(SiLU(xW_gate) * xW_up)."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism utilizing RoPE and fused CUDA kernels.

    Leverages PyTorch 2.0+ `F.scaled_dot_product_attention` (FlashAttention / Memory-Efficient Attention)
    to compute exact causal attention masks without materializing full $T \times T$ attention matrices in VRAM.
    """

    def __init__(self, config: GPT2Config):
        """Initializes linear query/key/value projections and rotary embedding helper.

        Args:
            config: Model architectural configuration dataclass.

        Raises:
            AssertionError: If hidden dimension width is not evenly divisible by head count.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Hidden dimension must be divisible by attention head count."
        
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Linear projections without bias for clean optimization landscape
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=config.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes multi-head causal attention forward pass.

        Args:
            x: Input activation tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Projected attention context representations of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Project input activations to query, key, and value representations
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Separate heads: reshape from (B, T, D) -> (B, T, n_head, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # 3. Inject relative sequence positions via Rotary Embeddings
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # 4. Transpose head dimensions for SDPA matrix multiplication: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Execute hardware-fused Scaled Dot-Product Attention with causal masking
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=True
        )
        
        # 6. Concatenate attention heads and project back to hidden dimension space
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_out)


class GPT2Block(nn.Module):
    """Transformer processing block following modern Pre-Layer Normalization topology.

    Pre-LN applies normalization prior to attention and feed-forward sub-layers, establishing
    unimpeded identity residual pathways across deep architectures for optimal gradient flow.
    """

    def __init__(self, config: GPT2Config):
        """Instantiates RMSNorm layers, causal attention sub-module, and SwiGLU FFN.

        Args:
            config: Architectural configuration dataclass.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ffn = SwiGLU(config.n_embd, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes residual block forward pass: x + SubLayer(Norm(x))."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT2Model(nn.Module):
    """Custom Autoregressive GPT-2 Small neural network for perplexity anomaly detection.

    Integrates weight tying between input vocabulary embeddings and the output language modeling head
    to reduce parameter footprint and regularize next-token distribution projections.
    """

    def __init__(self, config: GPT2Config):
        """Constructs embedding tables, stacked transformer blocks, and output projections.

        Args:
            config: Model architectural configuration dataclass.
        """
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: tie token embedding weights directly to language modeling output head
        self.token_embeddings.weight = self.lm_head.weight
        
        # Apply custom Gaussian standard deviation weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes parameters following Llama/GPT-2 academic distribution standards ($\sigma=0.02$)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Executes the full autoregressive forward pass and optional cross-entropy loss evaluation.

        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len) containing log token IDs.
            targets: Optional integer tensor of shifted target IDs for training loss evaluation.

        Returns:
            Tuple containing:
                - `logits`: Float tensor of shape (batch_size, seq_len, vocab_size).
                - `loss`: Scalar mean cross-entropy loss tensor if targets are provided, else None.
        """
        # 1. Map integer token IDs to continuous dense vector embeddings
        x = self.token_embeddings(input_ids)
        
        # 2. Iterate through stacked transformer processing blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Apply final terminal layer normalization and project to vocabulary logits
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten batch and sequence axes to evaluate standard token cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)
            )
            
        return logits, loss
