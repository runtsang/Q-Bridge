"""Quantum‑enhanced hybrid model combining QCNN feature extraction with transformer classification."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QCNNFeatureExtractor(nn.Module):
    """Same as in classical version but can be replaced with a quantum feature extractor."""
    def __init__(self, input_dim: int = 8, embed_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.Tanh())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x)

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

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑augmented attention that projects the token embeddings through a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for i in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[i, i+1])
            tqf.cnot(q_device, wires=[self.n_wires-1, 0])
            return self.measure(q_device)
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        outputs = []
        for head in range(self.num_heads):
            head_tokens = x[:, head]  # (batch, seq_len, d_k)
            flat = head_tokens.reshape(-1, self.d_k)
            q_device = self.q_device.copy(bsz=flat.size(0), device=flat.device)
            out = self.q_layer(flat, q_device)
            out = out.reshape(batch_size, seq_len, self.d_k)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return out
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._apply_quantum_heads(x)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        flat = x.reshape(-1, embed_dim)
        if embed_dim!= self.n_qubits:
            flat = flat[:, :self.n_qubits]
        q_device = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        out = self.q_layer(flat, q_device)
        out = out.reshape(batch_size, seq_len, self.n_qubits)
        out = self.linear1(self.dropout(out))
        out = self.linear2(F.relu(out))
        return out

class TransformerBlockQuantum(nn.Module):
    """Quantum transformer block."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TextClassifierQuantum(nn.Module):
    """Quantum‑enhanced transformer‑based classifier."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 n_qlayers: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim,
                n_qubits_transformer, n_qubits_ffn, n_qlayers,
                q_device=q_device,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

class QCNNTransformerHybrid(nn.Module):
    """Hybrid quantum model: QCNN feature extractor + quantum transformer classifier."""
    def __init__(self,
                 input_dim: int = 8,
                 embed_dim: int = 16,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 n_qlayers: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(input_dim, embed_dim)
        self.classifier = TextClassifierQuantum(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_dim=ffn_dim,
            num_classes=num_classes,
            dropout=dropout,
            n_qubits_transformer=n_qubits_transformer,
            n_qubits_ffn=n_qubits_ffn,
            n_qlayers=n_qlayers,
            q_device=q_device
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x.float())
        features = features.unsqueeze(1)
        return self.classifier(features)

__all__ = ["QCNNTransformerHybrid", "QCNNFeatureExtractor", "TextClassifierQuantum", "TransformerBlockQuantum"]
