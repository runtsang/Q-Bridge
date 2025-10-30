"""Quantum‑enhanced self‑attention transformer in PennyLane."""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

class QuantumSelfAttentionCircuit:
    """PennyLane QNode implementing a self‑attention‑style circuit."""
    def __init__(self, n_qubits: int, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)

    def _rotations(self, params: np.ndarray):
        for i in range(self.n_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)

    def _entangle_all(self):
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])

    def __call__(self, params: np.ndarray):
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            self._rotations(params)
            self._entangle_all()
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit()

class QuantumSelfAttentionModel(nn.Module):
    """Hybrid model that uses the quantum attention circuit as a feature extractor."""
    def __init__(self, embed_dim: int, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.circuits = nn.ModuleList([QuantumSelfAttentionCircuit(n_qubits) for _ in range(n_layers)])
        self.linear = nn.Linear(n_qubits, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for token in x.unbind(dim=1):
            # Build rotation parameters: each embedding dimension becomes an RX angle,
            # other rotations set to zero.
            params = np.zeros((self.n_qubits, 3))
            params[:, 0] = token[:, :self.n_qubits].cpu().numpy()
            # Forward through the first circuit (all layers could be chained)
            q_out = []
            for b in range(batch):
                q_out.append(self.circuits[0](params[b]))
            q_out = torch.stack(q_out, dim=0)  # (batch, n_qubits)
            out.append(q_out)
        out = torch.stack(out, dim=1)  # (batch, seq, n_qubits)
        out = self.linear(out)
        return out

class QuantumTextClassifier(nn.Module):
    """Simple classifier that combines a classical embedding, positional encoding
    and a quantum self‑attention layer."""
    def __init__(self, vocab_size: int, embed_dim: int, n_qubits: int, num_classes: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Parameter(torch.zeros(1, 512, embed_dim))
        self.quantum_attn = QuantumSelfAttentionModel(embed_dim, n_qubits)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = tokens + self.positional[:, :tokens.size(1)]
        x = self.quantum_attn(x)
        x = x.mean(dim=1)
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttentionCircuit",
    "QuantumSelfAttentionModel",
    "QuantumTextClassifier",
]
