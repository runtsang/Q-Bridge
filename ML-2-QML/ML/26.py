"""Hybrid convolutional filter combining classical Conv2d and a learnable quantum circuit."""

from __future__ import annotations

import torch
from torch import nn
import pennylane as qml
import pennylane.numpy as qnp

class ConvDual(nn.Module):
    """A hybrid filter that merges a classical conv2d with a PennyLane variational circuit."""

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 n_qubits: int | None = None,
                 device: str = "default.qubit",
                 n_layers: int = 2,
                 shared_params: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits or kernel_size ** 2
        self.device = device
        self.n_layers = n_layers
        self.shared_params = shared_params

        # Classical convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum circuit
        self.qcircuit = self._build_qcircuit()

        # Linear head
        self.head = nn.Linear(2, 1)

    def _build_qcircuit(self):
        """Build a PennyLane variational circuit."""
        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            # Encode input as rotation angles
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
            # Variational layers
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(qnp.random.randn(), wires=i)
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, x):
        """
        Forward pass over a batch of images.

        Args:
            x: Tensor of shape (batch, 1, H, W)

        Returns:
            Tensor of shape (batch, 1) containing the feature value
            for each image, averaged over all sliding windows.
        """
        batch, _, H, W = x.shape
        out = torch.zeros(batch, device=x.device)

        # Slide over windows
        for i in range(H - self.kernel_size + 1):
            for j in range(W - self.kernel_size + 1):
                patch = x[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                # Classical conv
                conv_out = self.conv(patch).view(batch)
                # Quantum
                patch_flat = patch.view(batch, -1).cpu().numpy()
                q_out = torch.tensor([self.qcircuit(torch.tensor(p)) for p in patch_flat], device=x.device)
                # Concatenate and head
                combined = torch.stack([conv_out, q_out], dim=1)
                out += self.head(combined).squeeze(1)
        out /= ((H - self.kernel_size + 1) * (W - self.kernel_size + 1))
        return out.unsqueeze(1)

def Conv():
    """Return a ConvDual instance with default parameters."""
    return ConvDual()

__all__ = ["ConvDual"]
