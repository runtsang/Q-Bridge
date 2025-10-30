"""
Hybrid quanvolution filter using PennyLane.

Each 2×2 image patch is encoded into a 4‑qubit register via Ry rotations,
passed through a parameter‑shared entangling layer, and the expectation
value of Pauli‑Z on each qubit is returned.  The circuit is executed on a
CPU simulator and is compatible with PyTorch autograd.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumQuanvolutionFilter(nn.Module):
    """
    Variational quantum filter that processes 2×2 image patches.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 2, device: str = "cpu") -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=1024, device=device)

        # Encoder weights for Ry rotations (shared across patches)
        self.encoder_weights = nn.Parameter(torch.randn(num_qubits))
        # Entangling layer weights (shared across patches)
        self.entangling_weights = nn.Parameter(torch.randn(num_layers, num_qubits, num_qubits))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # Encode pixel values into Ry rotations
            for i in range(num_qubits):
                qml.Ry(x[i], wires=i)
            # Entangling layers
            for l in range(num_layers):
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qml.CNOT(wires=[i, j])
                        qml.RZZ(self.entangling_weights[l, i, j], wires=[i, j])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Flattened feature vector of shape (B, 4 * 14 * 14).
        """
        B = x.shape[0]
        # Extract 2×2 patches: shape (B, 1, 14, 14, 2, 2)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        # Rearrange to (B, 196, 4)
        patches = patches.permute(0, 2, 3, 4, 5).reshape(B, 14 * 14, 4)

        outputs = []
        for i in range(196):
            patch = patches[:, i, :]  # (B, 4)
            out = self.circuit(patch).reshape(B, -1)  # (B, 4)
            outputs.append(out)
        # Concatenate all patch outputs
        return torch.cat(outputs, dim=1)


class QuantumQuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that stacks the quantum filter with a linear head.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuantumQuanvolutionClassifier"]
