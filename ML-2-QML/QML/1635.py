"""QuantumNATEnhanced: quantum model variant using PennyLane."""

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np

class QuantumNATEnhanced(nn.Module):
    """Quantum implementation of the Quantum‑NAT‑style architecture.
    The network encodes the image into four qubits via a simple
    feature‑to‑rotation mapping, processes them through a
    parameter‑efficient variational circuit, then measures
    Pauli‑Z on each qubit.  The output is batch‑normed to match
    the classical head.
    """

    def __init__(self, wires=4):
        super().__init__()
        self.wires = wires
        self.dev = qml.device("default.qubit", wires=self.wires)

        # Variational parameters (3 layers, one per qubit)
        self.params = nn.Parameter(torch.randn(3, self.wires))
        # Linear projection from pooled features (16) to qubit count
        self.proj = nn.Linear(16, self.wires, bias=False)

        # QNode definition
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # Batch‑normalization for the 4‑dimensional output
        self.norm = nn.BatchNorm1d(self.wires)

    def _circuit(self, params, features):
        """Circuit that encodes the input features and runs the ansatz."""
        # Encode each feature into a rotation around Y
        for idx, f in enumerate(features):
            qml.RY(np.pi * f, wires=idx)

        # Apply 3 layers of the ansatz
        for layer_idx in range(3):
            for w in range(self.wires):
                qml.RX(params[layer_idx, w], wires=w)
            # Entangling chain
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measure Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, H, W) on the same device as the model.
        Returns:
            Tensor of shape (B, 4) after batch‑normalization.
        """
        bsz = x.shape[0]
        # Global average pool to 16 features
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        # Reduce to 4 features by linear projection
        projected = self.proj(pooled)

        out_batch = []
        for i in range(bsz):
            out = self.qnode(self.params, projected[i])
            out_batch.append(out)
        out = torch.stack(out_batch, dim=0)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
