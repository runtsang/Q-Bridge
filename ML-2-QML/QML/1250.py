import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np

class QuanvolutionNet(nn.Module):
    """
    Hybrid classical‑quantum implementation of a quanvolutional network.
    Uses a parameterised 4‑qubit variational circuit per 2×2 image patch.
    The circuit is trained end‑to‑end with PyTorch autograd.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        device: str = "cpu",
        wires: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = nn.Dropout(dropout)
        # Quantum device
        self.q_dev = qml.device("default.qubit", wires=wires)
        # Variational parameters
        self.var_params = nn.Parameter(
            torch.randn(num_layers, wires, 3)
        )  # rotation angles for RX, RZ, RY
        # Scaling factor for measurement outputs
        self.scale = nn.Parameter(torch.ones(1))
        # Linear classifier
        self.fc = nn.Linear(out_channels * 14 * 14, num_classes)

        # Quantum node
        @qml.qnode(self.q_dev, interface="torch", diff_method="backprop")
        def circuit(p, data):
            # Encode pixel values as rotations around Y
            for i in range(wires):
                qml.RY(data[i], wires=i)
            # Variational layers
            for layer in range(num_layers):
                for i in range(wires):
                    qml.RX(p[layer, i, 0], wires=i)
                    qml.RZ(p[layer, i, 1], wires=i)
                    qml.RY(p[layer, i, 2], wires=i)
                # Entanglement
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[wires - 1, 0])
            # Measure expectation values of Z
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Tensor of shape (B, C, H, W) with values in [0,1].
        :return: Log‑softmax logits of shape (B, num_classes).
        """
        batch_size = x.size(0)
        # Extract non‑overlapping 2x2 patches
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # patches shape: (B, C, 14, 14, 2, 2)
        patches = patches.contiguous().view(batch_size, 14 * 14, 4)
        # Run quantum circuit for each patch
        features = []
        for i in range(patches.size(1)):
            data = patches[:, i, :]  # shape (B, 4)
            out = self.circuit(self.var_params, data)  # shape (B, 4)
            features.append(out)
        # Concatenate features from all patches
        x = torch.cat(features, dim=1)  # shape: (B, 14*14*4)
        x = self.scale * x
        x = self.dropout(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
