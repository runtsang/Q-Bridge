"""Hybrid self‑attention classifier with quantum‑derived attention."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import qiskit
from qiskit import Aer, execute

class ClassicalSelfAttention(nn.Module):
    """Differentiable self‑attention module with optional multi‑head support."""
    def __init__(self, embed_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_lin = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(-1, self.n_heads, self.head_dim)
        v = v.view(-1, self.n_heads, self.head_dim)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(scores, v)
        return self.out_lin(out.view(-1, self.embed_dim))

class QuantumAttentionWrapper(nn.Module):
    """Wraps a parameterised Qiskit circuit that yields a weight vector."""
    def __init__(self, n_qubits: int, depth: int = 1, backend=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qr = qiskit.QuantumRegister(self.n_qubits)
        cr = qiskit.ClassicalRegister(self.n_qubits)
        qc = qiskit.QuantumCircuit(qr, cr)
        for d in range(self.depth):
            for i in range(self.n_qubits):
                qc.rx(np.random.rand(), i)
                qc.ry(np.random.rand(), i)
                qc.rz(np.random.rand(), i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(qr, cr)
        return qc

    def forward(self, batch_size: int, shots: int = 1024) -> torch.Tensor:
        job = execute(self.circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = cnt / shots
        return torch.tensor(probs, dtype=torch.float32, device='cpu')

class FeedForwardClassifier(nn.Module):
    """Simple feed‑forward classifier with a configurable depth."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, depth: int = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridSelfAttention(nn.Module):
    """Hybrid self‑attention model that fuses classical attention with quantum‑derived weights."""
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        n_qubits: int,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.depth = depth

        self.attn = ClassicalSelfAttention(embed_dim, n_heads=n_heads)
        self.q_attn = QuantumAttentionWrapper(n_qubits, depth=depth)
        self.encoder = nn.Linear(1, embed_dim, bias=False)
        self.classifier = FeedForwardClassifier(embed_dim, hidden_dim, num_classes, depth=depth)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = inputs.shape
        encoded = self.encoder(inputs.float())
        attn_out = self.attn(encoded)
        q_weights = self.q_attn(batch_size)
        fused = attn_out * q_weights.unsqueeze(0).repeat(batch_size, 1)
        logits = self.classifier(fused)
        return logits

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, num_classes={self.num_classes}, n_qubits={self.q_attn.n_qubits}, depth={self.depth}"
