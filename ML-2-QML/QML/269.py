"""QuantumHybridClassifier – Pennylane quantum head with parameter‑shared circuit."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

__all__ = ["QuantumHybridClassifierQML", "PennylaneQuantumHead"]

class PennylaneQuantumHead(nn.Module):
    """Two‑qubit variational circuit measuring σ_z on qubit 0."""
    def __init__(self,
                 n_qubits: int = 2,
                 device: str = "default.qubit",
                 shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Parameter‑shared rotation on each qubit
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        # Expectation of PauliZ on qubit 0
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad or truncate to match n_qubits
        params = torch.cat([x, torch.ones((x.shape[0], self.n_qubits - x.shape[1]))], dim=1)
        return self.qnode(params, x)

class QuantumHybridClassifierQML(nn.Module):
    """Hybrid classifier that couples a classical MLP with a Pennylane quantum head."""
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 n_qubits: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.mlp = self._build_mlp()
        self.quantum_head = PennylaneQuantumHead(n_qubits)

    def _build_mlp(self) -> nn.Sequential:
        layers = []
        in_f = self.input_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(in_f, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            in_f = hidden
        layers.append(nn.Linear(in_f, self.n_qubits))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        q_out = self.quantum_head(y).unsqueeze(-1)
        return torch.cat((q_out, 1 - q_out), dim=-1)

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module) -> float:
        self.train()
        optimizer.zero_grad()
        logits = self(batch["input"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        return loss.item()
