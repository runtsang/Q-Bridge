"""Quantum model: 4‑qubit variational circuit with a classical encoder and linear read‑out."""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# 4‑qubit device
dev = qml.device("default.qubit", wires=4)

def _encoder(inputs: torch.Tensor, wires: list[int]) -> None:
    """Encode classical inputs into rotation angles on the qubits."""
    for i, w in enumerate(wires):
        qml.RY(inputs[:, i], wires=w)

def _variational(params: torch.Tensor, wires: list[int]) -> None:
    """A shallow entangling layer with trainable rotations."""
    for i, w in enumerate(wires):
        qml.RX(params[i], wires=w)
    # Entangling pattern
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[1, 2])
    for i, w in enumerate(wires):
        qml.RZ(params[4 + i], wires=w)

@qml.qnode(dev, interface="torch")
def _circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Quantum circuit that outputs expectation values of Pauli‑Z."""
    _encoder(inputs, wires=[0, 1, 2, 3])
    _variational(params, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QFCModelExtended(nn.Module):
    """
    Extended quantum model: a 4‑qubit variational circuit with a classical encoder
    and a linear read‑out head that maps the measurement vector to four logits.
    """
    def __init__(self, num_classes: int = 4, img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.num_params = 8  # 4 RX + 4 RZ
        self.params = nn.Parameter(torch.randn(self.num_params))
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) with H=W=28.

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional logits of shape (B, 4).
        """
        bsz = x.shape[0]
        flat = x.view(bsz, -1)                                   # (B, H*W)
        # Select four pixel indices evenly spaced across the flattened image
        idx = torch.linspace(0, flat.shape[1] - 1, steps=4, dtype=torch.long, device=x.device)
        angles = flat[:, idx] / 4.0                               # normalise to [0,1]
        angles = angles.to(torch.float)
        # Run the quantum circuit on the batch
        out = _circuit(angles, self.params)                      # (B, 4)
        out = self.mlp(out)                                      # (B, 4)
        return self.norm(out)

__all__ = ["QFCModelExtended"]
