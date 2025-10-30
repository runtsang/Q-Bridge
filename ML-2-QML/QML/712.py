"""Quantum quanvolution filter implemented with Pennylane.

The module mirrors the classical API but replaces the 2×2 convolution
with a variational circuit that encodes each 2×2 image patch into
four qubits.  The circuit contains a trainable parameter matrix
(`self.params`) and a fixed entangling pattern.  The measurement
outputs the expectation values of Pauli‑Z on each qubit, producing
a 4‑dimensional feature vector per patch.

The design is fully compatible with the original ``QuanvolutionFilter``
and ``QuanvolutionClassifier`` classes, so the new quantum module can
be dropped into existing pipelines without modification.

Example
-------
>>> from Quanvolution__gen382 import QuanvolutionFilter, QuanvolutionClassifier
>>> model = QuanvolutionClassifier()
>>> dummy = torch.randn(4, 1, 28, 28)
>>> out = model(dummy)
>>> out.shape
torch.Size([4, 10])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """Variational quanvolution filter using a 4‑qubit circuit.

    Args:
        n_qubits: Number of qubits per patch (default 4 for 2×2 patches).
        n_layers: Depth of the variational ansatz.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Parameter matrix of shape (n_layers, n_qubits, 3) for RX,RZ,RX rotations
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=0)

        @qml.qnode(self.dev, interface="torch")
        def _qnode(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the 4‑pixel patch into the qubits with RY rotations
            for i in range(n_qubits):
                qml.RY(x[:, i], wires=i)
            # Variational layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(params[l, i, 0], wires=i)
                    qml.RZ(params[l, i, 1], wires=i)
                    qml.RX(params[l, i, 2], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return [
                qml.expval(qml.PauliZ(i)) for i in range(n_qubits)
            ]

        self._qnode = _qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r : r + 2, c : c + 2].view(bsz, -1)
                measurement = self._qnode(patch, self.params)
                # measurement is a tuple of tensors; stack them
                patches.append(torch.stack(measurement, dim=1))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that stacks the quantum quanvolution filter
    with a linear head, exactly mirroring the original public API.

    Args:
        n_qubits: Number of qubits per patch.
        n_layers: Depth of the variational ansatz.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_qubits=n_qubits, n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
