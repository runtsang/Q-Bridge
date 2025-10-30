import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention that keeps the embedding and head size."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Straightforward multi‑head attention with PyTorch’s MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, x, x, key_padding_mask=mask)[0]


class _QuantumProjection(tq.QuantumModule):
    """Quantum module that projects a vector via a small parameterized circuit."""

    def __init__(self, dim: int, q_device: tq.QuantumDevice):
        super().__init__()
        self.dim = dim
        self.q_device = q_device
        # Encode each classical feature into an RX gate
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(dim)]
        )
        # Parameterised rotation around Y for each qubit
        self.r_y = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.r_y):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Multi‑head attention that replaces the linear projections with quantum modules.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        q_device: Optional[tq.QuantumDevice] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.head_dim)
        self.q_proj = nn.ModuleList(
            [_QuantumProjection(self.head_dim, self.q_device) for _ in range(num_heads)]
        )
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.size()
        # Split into heads
        x = x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        # Project each head quantumly
        proj = []
        for h in range(self.num_heads):
            head = x[:, h, :, :]
            # use same device for all tokens in this head
            qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
            proj.append(self.q_proj[h](head, qdev))
        proj = torch.stack(proj, dim=1)  # (B, H, S, D)
        # Concatenate heads
        proj = proj.view(B, S, self.embed_dim)
        # Standard dot‑product attention
        scores = torch.matmul(proj, proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, proj)
        return self.combine(out)


class FeedForwardBase(nn.Module):
    """Base feed‑forward block interface."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class _QuantumFFN(tq.QuantumModule):
    """Quantum module that implements a simple feed‑forward sub‑network."""

    def __init__(self, dim: int, q_device: tq.QuantumDevice):
        super().__init__()
        self.dim = dim
        self.q_device = q_device
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(dim)]
        )
        self.r_y = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.r_y):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.q_layer = _QuantumFFN(n_qubits, self.q_device)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()
        out = []
        for i in range(S):
            qdev = self.q_device.copy(bsz=B, device=x.device)
            out.append(self.q_layer(x[:, i, :], qdev))
        out = torch.stack(out, dim=1)  # (B, S, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, ffn_dim, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, ffn_dim, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, q_device=tq.QuantumDevice(n_wires=n_qubits_transformer)
        )
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumEnhancedTransformer(nn.Module):
    """
    Transformer-based text classifier supporting quantum submodules.
    When `n_qubits_transformer` or `n_qubits_ffn` are zero the model uses
    purely classical blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0 or n_qubits_ffn > 0:
            blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer or embed_dim,
                    n_qubits_ffn or embed_dim,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
