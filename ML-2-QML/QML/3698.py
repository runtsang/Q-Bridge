"""Quantum implementation of ConvGen345 using Pennylane.

The quantum filter replaces the classical convolution with a variational
circuit that encodes each image patch into a 4‑qubit state. The rest of the
network (CNN backbone + fully‑connected head) remains identical to the
classical version.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
import torch.nn.functional as F

# Quantum device – 4 qubits match the 2×2 kernel
dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_filter(patch: torch.Tensor) -> torch.Tensor:
    """
    Variational filter acting on a single image patch.

    Args:
        patch: Tensor of shape (batch, 4) with values in [0, π].

    Returns:
        Tensor of shape (batch,) containing the mean Pauli‑Z expectation.
    """
    # Encode pixel values into rotation angles
    for i in range(patch.shape[1]):
        qml.RY(patch[:, i], wires=i)

    # Small entangling layer
    if patch.shape[1] >= 2:
        for i in range(0, patch.shape[1] - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    # Compute mean Pauli‑Z expectation over all qubits
    z_expect = torch.stack(
        [qml.expval(qml.PauliZ(wires=i)) for i in range(patch.shape[1])], dim=1
    )
    return torch.mean(z_expect, dim=1)


class ConvGen345(nn.Module):
    """
    Quantum surrogate for a convolutional neural network.
    Uses a variational filter applied to image patches, followed by
    a classical CNN backbone and fully‑connected head.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_out_channels: int = 8,
        conv_stride: int = 1,
        conv_padding: int = 0,
        threshold: float = 0.0,
        fc_hidden: int = 64,
        fc_out: int = 4,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # For a 2×2 kernel we need 4 qubits
        self.n_qubits = kernel_size ** 2

        # CNN backbone identical to the classical version
        self.features = nn.Sequential(
            nn.Conv2d(conv_out_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_out),
        )
        self.norm = nn.BatchNorm1d(fc_out)

    def _quantum_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum filter to every 2×2 patch of the input image.
        """
        # Extract patches: shape (batch, n_qubits, H*W)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.kernel_size)
        # Re‑shape to (batch, patches, n_qubits)
        patches = patches.permute(0, 2, 1)
        batch, patches_cnt, n_qubits = patches.shape

        # Map pixel values to rotation angles in [0, π]
        angles = (patches / 255.0) * torch.pi
        angles = angles.reshape(batch * patches_cnt, n_qubits)

        # Run the variational circuit in a single batch
        outputs = quantum_filter(angles)
        outputs = outputs.reshape(batch, patches_cnt)

        # Reshape back to spatial dimensions
        out = outputs.reshape(
            x.shape[0],
            1,
            x.shape[2] // self.kernel_size,
            x.shape[3] // self.kernel_size,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the quantum convolution.
        """
        q = self._quantum_conv(x)
        features = self.features(q)
        flattened = features.view(features.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["ConvGen345"]
