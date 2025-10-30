"""Hybrid transformer with optional quantum modules for classical training."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------- Quantum submodules --------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑aware multi‑head attention using a small variational circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._build_q_layer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k, bsz=1)

    def _build_q_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, d_k: int) -> None:
                super().__init__()
                self.n_wires = d_k
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(d_k)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for gate in self.parameters:
                    gate(q_device)
                return self.measure(q_device)

        return QLayer(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Linear projections (classical)
        linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(x.device)
        k = linear(x)
        q = linear(x)
        v = linear(x)
        # Quantum projection
        batch, seq_len, _ = x.shape
        k_q = self._quantum_project(k)
        q_q = self._quantum_project(q)
        v_q = self._quantum_project(v)
        # Scaled dot‑product attention
        scores = torch.matmul(q_q, k_q.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_q)
        out = linear(out)  # combine heads
        return out

    def _quantum_project(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the quantum module per head
        seq_len = x.size(1)
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :].view(x.size(0), self.num_heads, self.d_k)
            token_out = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                token_out.append(self.q_layer(head, qdev))
            outputs.append(torch.stack(token_out, dim=1))
        return torch.stack(outputs, dim=1)

class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward network with a variational layer."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self._build_q_layer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_q_layer(self, n_qubits: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_qubits: int) -> None:
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for gate in self.parameters:
                    gate(q_device)
                return self.measure(q_device)

        return QLayer(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # Apply quantum layer to each token
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :].view(x.size(0), -1)
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------- Classical submodules --------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------- Transformer block --------------------- #
class TransformerBlockHybrid(nn.Module):
    """A transformer block that can be quantum or classical."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 n_qubits_attention: int = 8,
                 n_qubits_ffn: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits_attention)
                     if use_quantum_attn else MultiHeadAttentionClassical(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
                     if use_quantum_ffn else FeedForwardClassical(embed_dim, ffn_dim, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------- Positional encoding --------------------- #
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

# --------------------- Estimator head --------------------- #
class EstimatorHead(nn.Module):
    """Simple regression head that can be replaced by a quantum EstimatorQNN."""
    def __init__(self, embed_dim: int, use_quantum: bool = False, n_qubits: int = 8) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            # Quantum estimator using a 1‑qubit circuit
            self.q_layer = self._build_q_estimator(n_qubits)
        else:
            self.linear = nn.Linear(embed_dim, 1)

    def _build_q_estimator(self, n_qubits: int) -> tq.QuantumModule:
        class QEstimator(tq.QuantumModule):
            def __init__(self, n_qubits: int) -> None:
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for gate in self.parameters:
                    gate(q_device)
                return self.measure(q_device)

        return QEstimator(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            batch = x.size(0)
            # Flatten token dimension for quantum evaluation
            flat = x.view(batch, -1)
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=flat.size(0), device=flat.device)
            out = self.q_layer(flat, qdev)
            return out.mean(dim=1, keepdim=True)
        else:
            return self.linear(x.mean(dim=1))

# --------------------- Hybrid Transformer --------------------- #
class HybridTransformerEstimator(nn.Module):
    """A transformer that can use classical or quantum sub‑modules and a quantum estimator head."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 n_qubits_attention: int = 8,
                 n_qubits_ffn: int = 8,
                 use_quantum_estimator: bool = False,
                 n_qubits_estimator: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockHybrid(embed_dim, num_heads, ffn_dim,
                                   use_quantum_attn, use_quantum_ffn,
                                   n_qubits_attention, n_qubits_ffn,
                                   dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.estimator = EstimatorHead(embed_dim, use_quantum_estimator, n_qubits_estimator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.estimator(x)

__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "EstimatorHead",
    "HybridTransformerEstimator",
]
