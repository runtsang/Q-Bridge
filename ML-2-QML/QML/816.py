"""Quanvolutional filter inspired by ``quanvolution.py`` in the raw dataset."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """Apply a trainable 4‑qubit variational kernel to 2×2 image patches."""
    def __init__(self, n_qubits: int = 4, n_entanglements: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        # Rotation angles shared across all patches
        self.theta = nn.Parameter(torch.randn(n_qubits))
        # Entanglement pattern
        self.entanglements = [(i, i + 1) for i in range(n_entanglements)]

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Quantum node
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, theta: torch.Tensor):
            # Encode the 4 pixel values into rotations
            for i in range(n_qubits):
                qml.RY(theta[i] * x[i], wires=i)
            # Entanglement
            for (a, b) in self.entanglements:
                qml.CNOT(wires=[a, b])
            # Measurements: expectation of PauliZ
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, 1, 28, 28)
        Returns: Tensor of shape (B, 4*14*14)
        """
        bsz = x.size(0)
        patches = []
        # iterate over 2×2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # shape (B, 4)
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # apply quantum circuit to each sample in batch
                out = self.circuit(patch, self.theta)  # shape (B, 4)
                patches.append(out)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
