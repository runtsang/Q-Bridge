"""Hybrid quantum‑classical model using a parameterized variational circuit.

The QML version uses Pennylane’s differentiable quantum device.  For
every 2×2 patch of the input image a 4‑qubit variational circuit
(`StronglyEntanglingLayers`) produces a 4‑dimensional feature vector.
The concatenated feature map is fed into a classical linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionClassifier(nn.Module):
    """
    Quantum‑classical network that replaces the random quanvolution
    kernel with a trainable variational circuit per image patch.
    """

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.n_wires = 4

        # Parameters for the variational circuit
        self.weights = nn.Parameter(
            torch.randn(num_layers, self.n_wires, 3)
        )

        # Classical linear head
        self.linear = nn.Linear(self.n_wires * 14 * 14, 10)

        # Quantum device
        self.device = qml.device("default.qubit", wires=self.n_wires)

        # QNode that processes a single 4‑dimensional input vector
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantum circuit for a single 2×2 patch.

            Parameters
            ----------
            inputs : torch.Tensor
                Tensor of shape (4,) containing pixel values.

            Returns
            -------
            torch.Tensor
                Expectation values of PauliZ on each wire (shape (4,)).
            """
            for i in range(self.n_wires):
                qml.RY(inputs[i], wires=i)

            qml.StronglyEntanglingLayers(self.weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        batch = x.shape[0]

        # Extract 2×2 patches via unfold
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(batch, 14 * 14, 4)

        # Flatten batch and patch dimensions for batched quantum evaluation
        patch_vectors = patches.view(-1, 4)  # (batch*14*14, 4)

        # Evaluate the circuit for all patches in parallel
        features = self.circuit(patch_vectors)  # (batch*14*14, 4)

        # Reshape back to image‑like feature map
        features = features.view(batch, 14 * 14, 4)
        features = features.view(batch, -1)  # (batch, 4*14*14)

        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionClassifier"]
