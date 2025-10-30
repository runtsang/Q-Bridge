import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np
from typing import Optional

class QFCModel(nn.Module):
    """Hybrid quantum-classical model using a variational circuit."""
    def __init__(self,
                 output_dim: int = 4,
                 n_qubits: int = 4,
                 n_layers: int = 3,
                 use_regression: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_regression = use_regression
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.qcircuit = qml.QNode(self._circuit, self.device,
                                  interface="torch",
                                  diff_method="backprop")
        # Encode classical features into qubit angles
        self.encoder = nn.Linear(16, n_qubits)
        # Readout classifier
        self.classifier = nn.Linear(n_qubits, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)
        self.loss_fn = nn.MSELoss() if use_regression else nn.CrossEntropyLoss()
        # Variational parameters
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )
    def _circuit(self, params: torch.Tensor, z: torch.Tensor):
        """Variational circuit with layer-wise rotations and entanglement."""
        for i in range(self.n_qubits):
            qml.RX(z[i], wires=i)
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.Rot(params[layer, wire, 0],
                        params[layer, wire, 1],
                        params[layer, wire, 2],
                        wires=wire)
            for wire in range(self.n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        z = self.encoder(pooled)  # shape (bsz, n_qubits)
        # Compute expectation values batch-wise
        out = torch.stack([self.qcircuit(self.params, z[i]) for i in range(bsz)], dim=0)
        out = self.classifier(out)
        out = self.norm(out)
        return out

__all__ = ["QFCModel"]
