"""Quantum‑enhanced hybrid classifier.

This module implements a binary classifier that can be instantiated with
either a CNN or a transformer backbone.  The final head is a parameterised
quantum circuit that returns an expectation value.  The design combines
the CNN+quantum expectation head seed with the transformer seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
#  Learnable activation
# --------------------------------------------------------------------------- #
class HybridActivation(nn.Module):
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

# --------------------------------------------------------------------------- #
#  Classical transformer backbone
# --------------------------------------------------------------------------- #
class TextTransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)  # global average pooling
        return self.dropout(x)

# --------------------------------------------------------------------------- #
#  Quantum expectation head
# --------------------------------------------------------------------------- #
class QuantumExpectationHead(nn.Module):
    """Two‑qubit circuit returning the expectation of Z."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.barrier()
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        thetas = theta.detach().cpu().numpy()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def exp_z(counts):
            total = sum(counts.values())
            probs = np.array(list(counts.values())) / total
            states = np.array([int(s, 2) for s in counts.keys()])
            z_vals = 1 - 2 * states
            return np.sum(z_vals * probs)

        if isinstance(result, list):
            return torch.tensor([exp_z(r) for r in result], dtype=torch.float32)
        return torch.tensor([exp_z(result)], dtype=torch.float32)

# --------------------------------------------------------------------------- #
#  Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridQuantumBinaryClassifier(nn.Module):
    """Drop‑in replacement that can use a CNN or a transformer, with an optional
    quantum expectation head.
    """
    def __init__(
        self,
        backbone: str = "cnn",
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_quantum_head: bool = True,
        n_qubits: int = 2,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone == "cnn":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.5),
                nn.Flatten(),
                nn.Linear(55815, 120),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 1),
            )
        elif backbone == "transformer":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for transformer backbone")
            self.backbone = TextTransformerBackbone(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}")

        if use_quantum_head:
            self.head = QuantumExpectationHead(n_qubits=n_qubits)
        else:
            self.head = HybridActivation(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return probs

__all__ = ["HybridQuantumBinaryClassifier"]
