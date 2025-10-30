import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention with linear projections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        qk = self.separate_heads(q)
        vk = self.separate_heads(v)
        scores = torch.einsum("bhqd,bhkd->bhqk", qk, vk) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, vk)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)


class QuantumAttentionLayer(tq.QuantumModule):
    """Quantum module that maps an embedding vector to another."""
    def __init__(self, embed_dim: int, n_wires: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear_out = nn.Linear(n_wires, embed_dim)

    def _forward_q(self, token_slice: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, token_slice)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        out = self.measure(q_device)
        return self.linear_out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embed_dim = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            token_slice = token[:, :self.n_wires] if self.n_wires <= token.shape[1] else torch.cat(
                [token, torch.zeros(batch, self.n_wires - token.shape[1], device=token.device)], dim=1
            )
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=token.device)
            out = self._forward_q(token_slice, qdev)
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention combining classical and quantum paths."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.classical = MultiHeadAttentionClassical(embed_dim, num_heads, dropout, use_bias)
        self.quantum_layer = QuantumAttentionLayer(embed_dim, n_wires)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        classical_out = self.classical(x, mask)
        quantum_out = self.quantum_layer(classical_out)
        gate = torch.sigmoid(self.gate)
        return gate * quantum_out + (1 - gate) * classical_out


class FeedForwardBase(nn.Module):
    """Base for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a shared quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def _forward_q(self, token_slice: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, token_slice)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

        def forward(self, token_slice: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            return self._forward_q(token_slice, q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            token_slice = token[:, :self.q_layer.n_qubits] if self.q_layer.n_qubits <= token.shape[1] else torch.cat(
                [token, torch.zeros(token.size(0), self.q_layer.n_qubits - token.shape[1], device=token.device)], dim=1
            )
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=token.size(0), device=token.device)
            q_out = self.q_layer(token_slice, qdev)
            outputs.append(q_out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class FeedForwardHybrid(FeedForwardBase):
    """Hybrid feed‑forward combining classical and quantum paths."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 0, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.classical = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.quantum = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout) if n_qubits > 0 else None
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical_out = self.classical(x)
        if self.quantum is not None:
            quantum_out = self.quantum(x)
            gate = torch.sigmoid(self.gate)
            return gate * quantum_out + (1 - gate) * classical_out
        else:
            return classical_out


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockHybrid(TransformerBlockBase):
    """Transformer block using hybrid attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires: int = 8, n_qubits: int = 0, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout, n_wires)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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
        return x + self.pe[:, :x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum submodules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_wires: int = 8,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockHybrid(embed_dim, num_heads, ffn_dim, n_wires, n_qubits, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
