"""Quantum‑enhanced quanvolution with trainable variational circuit.

The implementation encodes each 2×2 image patch into a 4‑qubit state,
applies a small parameterised circuit, and measures the Pauli‑Z
expectation values.  The resulting 4‑dimensional feature vector per
patch is concatenated into a flat representation that is fed to a
linear classifier.  The quantum kernel is fully differentiable and
its parameters are optimised together with the classical head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np


class Quanvolution__gen110(nn.Module):
    """Hybrid quantum‑classical quanvolutional network.

    The network processes an MNIST‑style image by extracting 2×2 patches,
    encoding each patch into a 4‑qubit circuit with trainable rotation
    angles, measuring the qubits, and then classifying the concatenated
    feature vector with a linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Quantum device with 4 wires
        self.dev = qml.device("default.qubit", wires=4)
        # Trainable parameters for the quantum circuit
        self.params = nn.Parameter(torch.randn(4))
        # Linear classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def _quantum_circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that returns a 4‑dimensional feature vector.

        Args:
            x: Tensor of shape (4,) containing pixel intensities of a
               2×2 patch, normalised to [0, 1].
            params: Tensor of shape (4,) with trainable rotation angles.
        Returns:
            Tensor of shape (4,) with expectation values of Pauli‑Z.
        """
        @qml.qnode(self.dev, interface="torch")
        def circuit(data: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            # Data encoding (Ry rotations)
            for i in range(4):
                qml.RY(data[i], wires=i)
            # Trainable rotations
            for i in range(4):
                qml.RY(theta[i], wires=i)
            # Entangling layer
            for i in range(3):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        return circuit(x, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid quanvolutional network.

        Args:
            x: Tensor of shape (batch, 1, 28, 28) with pixel values in [0, 1].
        Returns:
            Log‑softmax logits of shape (batch, 10).
        """
        bsz = x.shape[0]
        # Ensure pixel values are normalised to [0, 1]
        x = x.view(bsz, 28, 28)
        # Extract non‑overlapping 2×2 patches
        patches = torch.nn.functional.unfold(
            x.unsqueeze(1), kernel_size=2, stride=2
        )  # (bsz, 4*14*14)
        patches = patches.view(bsz, 4, 14 * 14)  # (bsz, 4, 196)
        patches = patches.permute(0, 2, 1)  # (bsz, 196, 4)
        # Process each patch with the quantum circuit
        quantum_features = []
        for i in range(patches.shape[1]):
            patch = patches[:, i, :]  # (bsz, 4)
            # Apply the circuit to each sample in the batch
            out = torch.stack(
                [self._quantum_circuit(patch[j], self.params) for j in range(bsz)],
                dim=0,
            )
            quantum_features.append(out)
        # Concatenate features from all patches
        quantum_features = torch.stack(quantum_features, dim=1)  # (bsz, 196, 4)
        quantum_features = quantum_features.view(bsz, -1)  # (bsz, 4*196)
        # Classifier head
        logits = self.classifier(quantum_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen110"]
