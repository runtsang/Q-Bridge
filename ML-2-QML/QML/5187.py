"""Hybrid binary classifier with quantum transformer blocks and a
quantum expectation head.  The public API matches the classical
variant so that the same class name can be imported from either
module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enabled multi‑head attention using a small quantum
    module per head.  The implementation follows the style of the
    torchquantum example in the reference pair."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = self.QLayer(n_wires=self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.size()
        if embed_dim % self.num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        # split heads
        x = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        # compute QN for each head
        q_outputs = []
        for head in range(self.num_heads):
            head_x = x[:, head]  # (batch, seq_len, d_k)
            # flatten seq_len for quantum processing
            seq_flat = head_x.reshape(-1, self.d_k)  # (batch*seq_len, d_k)
            qdev = self.q_device.copy(bsz=seq_flat.size(0), device=seq_flat.device)
            q_out = self.q_layer(seq_flat, qdev)  # (batch*seq_len, d_k)
            q_out = q_out.reshape(batch, seq_len, self.d_k)
            q_outputs.append(q_out)
        q_concat = torch.stack(q_outputs, dim=1).transpose(1, 2)  # (batch, seq_len, embed_dim)
        # simple scaled dot‑product attention using q outputs as keys/values
        # here we just use them as linear projections for demonstration
        q = self.out_proj(q_concat)
        k = self.out_proj(q_concat)
        v = self.out_proj(q_concat)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        out = self.drop(out)
        return out

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a small quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.size()
        out = []
        for i in range(seq_len):
            token = x[:, i]  # (batch, embed_dim)
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            q_out = self.q_layer(token, qdev)  # (batch, n_qubits)
            out.append(q_out)
        q_out = torch.stack(out, dim=1)  # (batch, seq_len, n_qubits)
        q_out = self.linear1(self.drop(q_out))
        return self.linear2(F.relu(q_out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop(ffn_out))

class QuantumHybridHead(nn.Module):
    """Quantum head that maps the aggregated embedding to a single
    probability via expectation of Pauli‑Z."""
    class QCircuit(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, n_qubits: int = 4, shift: float = 0.0) -> None:
        super().__init__()
        self.q_circuit = self.QCircuit(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim) -> we project to n_qubits
        proj = nn.Linear(x.size(1), self.q_circuit.n_qubits)(x)
        qdev = self.q_device.copy(bsz=proj.size(0), device=proj.device)
        exp = self.q_circuit(proj, qdev)  # (batch, n_qubits)
        # expectation of Z on all qubits and average
        exp_val = exp.mean(dim=-1)
        return torch.sigmoid(exp_val + self.shift)

class HybridBinaryClassifier(nn.Module):
    """Full quantum‑enabled binary classifier that mirrors the
    classical counterpart.  The interface is identical so that the
    same class name can be imported from either module."""
    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 n_qubits_ffn: int = 4,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.encoder = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_qubits_ffn) for _ in range(num_layers)]
        )
        self.head = QuantumHybridHead(n_qubits=n_qubits_ffn, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, 2) with class probabilities.
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # transformer expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1).mean(dim=1)
        prob = self.head(x)
        return torch.stack([prob, 1 - prob], dim=-1)

__all__ = ["HybridBinaryClassifier", "FraudLayerParameters"]
