"""Quantum‑enabled implementation of HybridTransformerClassifier.
It replaces the transformer blocks and convolutional filter with quantum modules
while keeping the same public API."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

# Quantum convolutional filter
def Conv():
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""
        def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = qiskit.execute(
                self._circuit, self.backend,
                shots=self.shots, parameter_binds=param_binds
            )
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return QuanvCircuit(filter_size=2, backend=backend, shots=100, threshold=127)

# Quantum transformer primitives
class MultiHeadAttentionQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # split heads
        q = k = v = x
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # apply quantum layer per head
        out = []
        for head in range(self.num_heads):
            head_out = []
            for token in q[:, head, :, :]:
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1, device=token.device)
                head_out.append(self.q_layer(token, qdev))
            out.append(torch.stack(head_out))
        out = torch.stack(out, dim=1).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.embed_dim)
        scores = torch.matmul(out, out.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, out)

class FeedForwardQuantum(nn.Module):
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
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_ffn: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class HybridTransformerClassifier(nn.Module):
    """Quantum‑enabled transformer classifier that mirrors the classical API.
    All parameters are identical to the classical version; only the internal
    implementation switches to quantum modules when `use_quantum=True`."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = True,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        # Quantum convolutional front‑end
        self.conv = Conv()
        # Transformer stack
        if self.use_quantum:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_ffn)
                    for _ in range(num_blocks)
                ]
            )
        else:
            # Fallback to classical blocks if quantum is disabled
            from torch.nn import MultiheadAttention, Linear, LayerNorm, Dropout
            class ClassicalBlock(nn.Module):
                def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
                    super().__init__()
                    self.norm1 = LayerNorm(embed_dim)
                    self.norm2 = LayerNorm(embed_dim)
                    self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                    self.ffn = nn.Sequential(
                        Linear(embed_dim, ffn_dim),
                        nn.ReLU(),
                        Dropout(dropout),
                        Linear(ffn_dim, embed_dim),
                    )
                    self.dropout = Dropout(dropout)

                def forward(self, x):
                    attn_out, _ = self.attn(x, x, x)
                    x = self.norm1(x + self.dropout(attn_out))
                    ffn_out = self.ffn(x)
                    return self.norm2(x + self.dropout(ffn_out))
            self.transformers = nn.Sequential(
                *[
                    ClassicalBlock(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume image data if 4‑D input; otherwise token indices
        if x.ndim == 4:
            x = self.conv.run(x.cpu().numpy())
            # Convert back to tensor; reshape to (batch, seq_len, 1)
            x = torch.tensor(x, dtype=torch.float32, device=x.device).unsqueeze(-1)
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "HybridTransformerClassifier",
]
