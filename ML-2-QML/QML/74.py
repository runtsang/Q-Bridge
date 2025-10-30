"""Quantum‑aware transformer with a quantum‑circuit‑based attention head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import math
from typing import Optional

from.QTransformerTorch import MultiHeadAttentionClassical, FeedForwardClassical

# --------------------------------------------------------------------------- #
# 1. Quantum projection module
# --------------------------------------------------------------------------- #
class QuantumProjection(tq.QuantumModule):
    """
    Encodes a classical vector into a quantum state, applies parametric gates,
    and measures all qubits. The output is a real‑valued vector of the same
    dimensionality as the number of wires.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Map each input dimension to a distinct qubit via an RX gate
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Trainable rotation angles
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        x: (B, n_wires)
        Returns: (B, n_wires)
        """
        self.encoder(q_device, *[x[:, i] for i in range(self.n_wires)])
        for wire, param in enumerate(self.params):
            tq.RX(q_device, wires=wire, params=param)
        return self.measure(q_device)

# --------------------------------------------------------------------------- #
# 2. Quantum‑aware attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantumWrapper(tq.QuantumModule):
    """Attention module that delegates Q, K, V projections to a quantum circuit."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits: int = 8,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = QuantumProjection(n_qubits)
        self.k_proj = QuantumProjection(n_qubits)
        self.v_proj = QuantumProjection(n_qubits)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)

    def _quantum_proj(self, x: torch.Tensor, proj: QuantumProjection) -> torch.Tensor:
        b, l, d = x.shape
        x_flat = x.view(b * l, d)
        qdev = self.q_device.copy(bsz=x_flat.size(0), device=x_flat.device)
        out = proj(x_flat, qdev)
        return out.view(b, l, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._quantum_proj(x, self.q_proj)
        k = self._quantum_proj(x, self.k_proj)
        v = self._quantum_proj(x, self.v_proj)
        q = q.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(out)

# --------------------------------------------------------------------------- #
# 3. Quantum feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_proj = QuantumProjection(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x_flat = x.view(b * l, d)
        qdev = self.q_device.copy(bsz=x_flat.size(0), device=x_flat.device)
        out = self.q_proj(x_flat, qdev)
        out = out.view(b, l, self.n_qubits)
        out = self.linear1(self.dropout(F.relu(out)))
        return self.linear2(out)

# --------------------------------------------------------------------------- #
# 4. Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockHybrid(tq.QuantumModule):
    """Hybrid transformer block that can use quantum attention/FFN."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        n_qubits: int = 8,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.attn = MultiHeadAttentionQuantumWrapper(
                embed_dim, num_heads, dropout, n_qubits, q_device
            )
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 5. Positional encoding (same as classical)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
# 6. Contrastive head
# --------------------------------------------------------------------------- #
class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, embed_dim: int, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# 7. Hybrid text classifier
# --------------------------------------------------------------------------- #
class TextClassifierHybrid(nn.Module):
    """Transformer‑based classifier that supports quantum blocks and contrastive head."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        share_weights: bool = False,
        use_quantum: bool = False,
        n_qubits: int = 8,
        q_device: tq.QuantumDevice | None = None,
        contrastive_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = (
            lambda: TransformerBlockHybrid(
                embed_dim,
                num_heads,
                ffn_dim,
                dropout,
                use_quantum,
                n_qubits,
                q_device,
            )
        )
        if share_weights:
            single_block = block_cls()
            blocks = nn.ModuleList([single_block] * num_blocks)
        else:
            blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.contrastive_head = (
            ContrastiveHead(embed_dim, contrastive_dim) if contrastive_dim is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        if self.contrastive_head is None:
            return logits
        return logits, self.contrastive_head(x)

__all__ = [
    "QuantumProjection",
    "MultiHeadAttentionQuantumWrapper",
    "FeedForwardQuantum",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "ContrastiveHead",
    "TextClassifierHybrid",
]
