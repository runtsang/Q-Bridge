import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionBase(nn.Module):
    """Base class for attention mechanisms."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise NotImplementedError


class MultiHeadAttentionClassical(_AttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)


class MultiHeadAttentionQuantumWrapper(nn.Module):
    """Wrapper that lazily imports the quantum implementation."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device=None):
        super().__init__()
        from.QTransformerTorch__gen338_qml import MultiHeadAttentionQuantum as QMA
        self.q_module = QMA(embed_dim, num_heads, dropout, q_device=q_device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.q_module(x, mask)


class _FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class FeedForwardClassical(_FeedForwardBase):
    """Standard two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class FeedForwardQuantumWrapper(nn.Module):
    """Wrapper that lazily imports the quantum implementation."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        from.QTransformerTorch__gen338_qml import FeedForwardQuantum as QFF
        self.q_module = QFF(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor):
        return self.q_module(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional learnable parameters."""
    def __init__(self, embed_dim: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.normal_(self.pe, std=0.02)
        else:
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float32)
                * -(math.log(10000.0) / embed_dim)
            )
            pe = torch.zeros(max_len, embed_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        if self.learnable:
            return x + self.pe[:, : x.size(1)]
        else:
            return x + self.pe[:, : x.size(1)]


class DropoutSchedule(nn.Module):
    """Dynamic dropout that decays from start to end over training steps."""
    def __init__(self, start: float = 0.1, end: float = 0.0, decay_steps: int = 10000):
        super().__init__()
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor):
        p = self.end + (self.start - self.end) * torch.exp(
            -self.step.float() / self.decay_steps
        )
        self.step += 1
        return F.dropout(x, p.item(), training=self.training)


class TransformerBlockHybrid(nn.Module):
    """Transformer block that can operate in classical, quantum, or mixed mode."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        mode: str = "classical",  # options: classical, quantum, mixed
        n_qubits_attention: int = 0,
        n_qubits_ffn: int = 0,
        q_device=None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in {"classical", "quantum", "mixed"}:
            raise ValueError("mode must be 'classical', 'quantum', or'mixed'")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if mode == "classical":
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        elif mode == "quantum":
            self.attn = MultiHeadAttentionQuantumWrapper(
                embed_dim, num_heads, dropout, q_device=q_device
            )
            self.ffn = FeedForwardQuantumWrapper(
                embed_dim, ffn_dim, n_qubits_ffn, dropout
            )
        else:  # mixed
            self.attn = MultiHeadAttentionQuantumWrapper(
                embed_dim, num_heads, dropout, q_device=q_device
            )
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TextClassifierHybrid(nn.Module):
    """Transformer‑based classifier with hybrid attention/FFN and configurable positional encoding."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        pos_learnable: bool = False,
        mode: str = "classical",
        n_qubits_attention: int = 0,
        n_qubits_ffn: int = 0,
        q_device=None,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, learnable=pos_learnable)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    mode=mode,
                    n_qubits_attention=n_qubits_attention,
                    n_qubits_ffn=n_qubits_ffn,
                    q_device=q_device,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def set_mode(self, mode: str):
        """Switch runtime mode of all blocks."""
        if mode not in {"classical", "quantum", "mixed"}:
            raise ValueError("mode must be 'classical', 'quantum', or'mixed'")
        for block in self.blocks:
            if isinstance(block, TransformerBlockHybrid):
                # Reinitialize attention and ffn according to new mode
                if mode == "classical":
                    block.attn = MultiHeadAttentionClassical(
                        block.attn.embed_dim, block.attn.num_heads, block.dropout.p
                    )
                    block.ffn = FeedForwardClassical(
                        block.ffn.embed_dim, block.ffn.ffn_dim, block.dropout.p
                    )
                elif mode == "quantum":
                    block.attn = MultiHeadAttentionQuantumWrapper(
                        block.attn.embed_dim,
                        block.attn.num_heads,
                        block.dropout.p,
                        q_device=block.attn.q_module.q_device,
                    )
                    block.ffn = FeedForwardQuantumWrapper(
                        block.ffn.embed_dim,
                        block.ffn.ffn_dim,
                        block.ffn.q_module.n_qubits,
                        block.dropout.p,
                    )
                else:  # mixed
                    block.attn = MultiHeadAttentionQuantumWrapper(
                        block.attn.embed_dim,
                        block.attn.num_heads,
                        block.dropout.p,
                        q_device=block.attn.q_module.q_device,
                    )
                    block.ffn = FeedForwardClassical(
                        block.ffn.embed_dim, block.ffn.ffn_dim, block.dropout.p
                    )

    def register_quantum_hook(self, hook_fn):
        """Register a hook to monitor quantum device usage."""
        for block in self.blocks:
            if hasattr(block.attn, "q_module"):
                block.attn.q_module.register_hook(hook_fn)

__all__ = [
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantumWrapper",
    "FeedForwardClassical",
    "FeedForwardQuantumWrapper",
    "TransformerBlockHybrid",
    "TextClassifierHybrid",
    "PositionalEncoding",
    "DropoutSchedule",
]
