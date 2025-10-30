"""
Quantum implementation of the Quanvolution hybrid network using Pennylane.
The filter is a parameter‑driven variational circuit applied to each 2×2 patch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionHybrid(nn.Module):
    """
    Quantum implementation of the Quanvolution hybrid network.

    The network consists of:
      * Extraction of non‑overlapping 2×2 patches.
      * A per‑patch variational quantum circuit with learnable parameters.
      * A linear classifier mapping the concatenated quantum features to class logits.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.num_layers = num_layers

        # Backend selection: use GPU if available, otherwise CPU.
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = qml.device("default.qubit", wires=4, shots=1024, device=backend)

        # Variational parameters: shape (num_layers, 4, 2)
        self.weights = nn.Parameter(torch.randn(num_layers, 4, 2, dtype=torch.float64))

        # Define the quantum node
        def _circuit(data: torch.Tensor, weights: torch.Tensor):
            # Encode the 2×2 patch into Ry rotations
            for i in range(4):
                qml.RY(data[i], wires=i)

            # Variational layers
            for l in range(self.num_layers):
                for q in range(4):
                    qml.RY(weights[l, q, 0], wires=q)
                    qml.RZ(weights[l, q, 1], wires=q)
                # Entanglement pattern (two CNOTs)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])

            # Measure expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.qnode = qml.QNode(
            _circuit,
            self.dev,
            interface="torch",
            diff_method="adjoint",
        )

        # Linear classifier
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (batch_size, num_classes).
        """
        B = x.size(0)

        # Extract non‑overlapping 2×2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape: (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(B, 14, 14, 4)
        patches = patches.view(B * 14 * 14, 4)  # shape: (B*14*14, 4)

        # Compute quantum features for each patch
        q_features = []
        for patch in patches:
            q_features.append(self.qnode(patch, self.weights))

        q_features = torch.stack(q_features)  # shape: (B*14*14, 4)
        q_features = q_features.view(B, 14, 14, 4)
        q_features = q_features.view(B, -1)  # flatten

        logits = self.classifier(q_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
