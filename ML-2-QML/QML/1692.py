"""
Hybrid quantum‑classical quanvolution that replaces the classical
2×2 convolution with a trainable variational quantum circuit.  The
circuit is defined using PennyLane and is fully differentiable
through the Torch interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumPatchLayer(nn.Module):
    """
    Variational circuit that processes a 2×2 image patch (4 values).
    The circuit consists of:
      * Ry encoding of the pixel values
      * A single layer of all‑to‑all CNOTs
      * Parameterised RZ rotations (trainable)
      * Expectation value of Z on each qubit
    """
    def __init__(self, wires: int = 4, device: str = "default.qubit") -> None:
        super().__init__()
        self.wires = wires
        self.dev = qml.device(device, wires=wires, shots=None)

        # Trainable rotation parameters
        self.rz_params = nn.Parameter(torch.randn(wires))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # inputs shape (batch, wires)
            for i in range(wires):
                qml.RY(inputs[:, i], wires=i)
            # All‑to‑all entanglement
            for i in range(wires):
                qml.CNOT(wires=[i, (i + 1) % wires])
            # Parameterised unitaries
            for i in range(wires):
                qml.RZ(self.rz_params[i], wires=i)
            # Measure Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        self.circuit = circuit

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: Tensor of shape (batch, num_patches, 4)
        Returns:
            Tensor of shape (batch, num_patches * 4)
        """
        batch, num_patches, _ = patches.shape
        outputs = []
        for i in range(num_patches):
            out = self.circuit(patches[:, i])
            outputs.append(out)
        return torch.cat(outputs, dim=1)

class QuantumQuanvolutionFilter(nn.Module):
    """
    Wrapper that extracts 2×2 patches from the input image and feeds
    them through the QuantumPatchLayer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.quantum_layer = QuantumPatchLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            Tensor of shape (B, 4*14*14)
        """
        batch = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, :, r : r + 2, c : c + 2].view(batch, -1)
                patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (B, 14*14, 4)
        features = self.quantum_layer(patches)
        return features

class QuantumQuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that combines the quantum quanvolution with a
    classical linear head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumPatchLayer", "QuantumQuanvolutionFilter", "QuantumQuanvolutionClassifier"]
