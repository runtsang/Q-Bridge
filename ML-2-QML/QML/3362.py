from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum libraries
import pennylane as qml
import pennylane.numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class QuantumSampler(nn.Module):
    """Variational sampler implemented with Qiskit."""
    def __init__(self, n_qubits: int = 2, n_params: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", n_params)
        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.ry(self.input_params[i], i)
        for i in range(n_qubits):
            self.circuit.ry(self.weight_params[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.cx(n_qubits - 1, 0)
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.input_params,
                              weight_params=self.weight_params,
                              sampler=self.sampler)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, 2)
        batch, seq, _ = inputs.shape
        probs = torch.zeros(batch, seq, 2, device=inputs.device)
        for b in range(batch):
            for s in range(seq):
                angles = inputs[b, s].cpu().numpy()
                probs[b, s] = torch.tensor(self.qnn.predict(angles), device=inputs.device)
        return probs


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForwardClassical(nn.Module):
    """Two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class QuantumAttention(nn.Module):
    """Quantum attention using a Pennylane variational circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dev: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dev = qml.device(dev, wires=self.num_heads * self.d_k)
        self.weights = nn.Parameter(torch.randn(2, self.num_heads * self.d_k))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        for i in range(self.num_heads * self.d_k):
            qml.RX(x[i], wires=i)
        qml.layer(qml.templates.StronglyEntanglingLayers, weights=weights)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_heads * self.d_k)]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq):
                flat = x[b, s].flatten()
                out[b, s] = self.qnode(flat, self.weights)
        return out


class QuantumTransformerBlock(nn.Module):
    """Transformer block with quantum attention and classical feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class UnifiedSamplerTransformer(nn.Module):
    """Quantum‑enhanced sampler‑transformer that uses the Qiskit sampler and Pennylane attention."""
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
        self.sampler = QuantumSampler()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = self.sampler(inputs)
        token_ids = torch.argmax(probs, dim=-1)
        x = self.embedding(token_ids)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["QuantumSampler", "QuantumAttention", "QuantumTransformerBlock",
           "UnifiedSamplerTransformer"]
