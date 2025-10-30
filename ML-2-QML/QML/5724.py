"""Quanvolutional filter using a Pennylane variational circuit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn import TorchLayer

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]

# Define a single‑shot variational circuit for a 2×2 patch
def _quantum_circuit(inputs, weights):
    """
    A simple variational circuit with 2 layers of Ry rotations and
    a CNOT entanglement pattern. `inputs` is a 4‑element array of
    pixel intensities; `weights` is a 2×4 array of trainable angles.
    """
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    for layer in range(2):
        qml.RY(weights[layer, 0], wires=0)
        qml.RY(weights[layer, 1], wires=1)
        qml.RY(weights[layer, 2], wires=2)
        qml.RY(weights[layer, 3], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Create a tensor network layer
dev = qml.device("default.qubit", wires=4)
qnn_layer = TorchLayer(_quantum_circuit, weight_shapes={"weights": (2, 4)})

class QuanvolutionFilter(nn.Module):
    """
    Applies the variational quantum circuit to each 2×2 patch of the input
    image. The output of the circuit is concatenated into a 1‑D feature
    vector that matches the dimensionality of the classical filter.
    """

    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.qnn = qnn_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = x.shape
        assert c == 1, "Only single‑channel images are supported."
        x = x.view(bsz, h, w)
        patches = []
        for r in range(0, h, 2):
            for col in range(0, w, 2):
                patch = x[:, r:r+2, col:col+2].reshape(bsz, -1)  # shape (bsz, 4)
                out = self.qnn(patch)  # shape (bsz, 4)
                patches.append(out)
        out = torch.cat(patches, dim=1)  # shape (bsz, 4 * (h/2)*(w/2))
        return out

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network that fuses the quantum feature map with a classical
    convolutional head and a linear classifier.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Classical convolutional head to mix the quantum features
        self.conv_head = nn.Conv1d(4 * 14 * 14, 64, kernel_size=1)
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.qfilter(x)  # shape (bsz, 4*14*14)
        x = self.conv_head(qfeat.unsqueeze(-1))  # shape (bsz, 64, 1)
        x = self.bn(x)
        x = torch.relu(x).squeeze(-1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)
