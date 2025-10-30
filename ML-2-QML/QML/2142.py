import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionBase(nn.Module):
    """
    Base class for multi‑head attention.  Mirrors the classical variant.
    """
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

class QuantumLinear(tq.QuantumModule):
    """
    Tiny variational circuit that maps an input vector to a quantum state,
    applies a parameterised rotation, and measures in the computational basis.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wires = min(input_dim, output_dim)
        # Encoding: each input dimension -> RX on corresponding wire
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
        )
        # Parameterised rotation
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, input_dim)
        # encode only up to n_wires
        self.encoder(q_device, x[..., :self.n_wires])
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        # add some entanglement
        for i in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[i, i + 1])
        # measure
        return self.measure(q_device)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Quantum‑enhanced multi‑head attention.  Each head is processed by a
    small variational circuit that replaces the linear projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = QuantumLinear(self.head_dim, self.head_dim)
        self.k_layer = QuantumLinear(self.head_dim, self.head_dim)
        self.v_layer = QuantumLinear(self.head_dim, self.head_dim)
        self.out_layer = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # split heads
        x_split = x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # process each head
        heads_out = []
        for head in range(self.num_heads):
            token = x_split[:, head]  # shape (batch, seq, head_dim)
            # flatten to (batch*seq, head_dim)
            token_flat = token.contiguous().view(-1, self.head_dim)
            qdev = tq.QuantumDevice(n_wires=self.head_dim, bsz=token_flat.size(0))
            out = self.q_layer(token_flat, qdev)
            out = out.view(batch, seq, self.head_dim)
            heads_out.append(out)
        q = torch.stack(heads_out, dim=1)  # (batch, heads, seq, head_dim)
        # Compute attention scores using classical dot product
        k = q  # for simplicity, use same as q
        v = q
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attn = torch.matmul(probs, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_layer(attn)

class FeedForwardBase(nn.Module):
    """
    Base class for feed‑forward layers.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    """
    Feed‑forward network realised by a small variational circuit.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.quantum_layer = QuantumLinear(embed_dim, ffn_dim)
        self.linear = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Flatten input
        x_flat = x.contiguous().view(-1, self.quantum_layer.input_dim)
        qdev = tq.QuantumDevice(n_wires=self.quantum_layer.n_wires, bsz=x_flat.size(0))
        qout = self.quantum_layer(x_flat, qdev)
        qout = qout.view(batch, seq, self.ffn_dim)
        return self.linear(self.dropout(F.relu(qout)))

class HybridTransformerBlockQuantum(nn.Module):
    """
    Transformer block that uses quantum attention and feed‑forward sub‑modules.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class QuantumHybridTransformer(nn.Module):
    """
    Text classifier built from quantum transformer blocks.  The API is the
    same as the classical seed, enabling side‑by‑side evaluation.
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
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[HybridTransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "HybridTransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumHybridTransformer",
]
