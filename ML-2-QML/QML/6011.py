"""Quanvolutional filter implemented with a variational quantum circuit using PennyLane."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np

class QuanvolutionHybridClassifier(nn.Module):
    """
    Quantum implementation of a quanvolutional filter followed by a linear classifier.
    Each 2x2 image patch is encoded into a 4-qubit circuit, processed by a variational ansatz,
    and the expectation values of Pauli-Z are used as patch features.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, patch_size: int = 2, stride: int = 2,
                 dev_name: str = "default.qubit", wires: int = 4, layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.dev = qml.device(dev_name, wires=wires)
        self.layers = layers
        # Parameter shape: (layers, wires, 3) for rotations around X, Y, Z
        self.params = nn.Parameter(torch.randn(layers, wires, 3))
        self.head = nn.Linear(4 * 14 * 14, num_classes)

        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            # Encode inputs: each pixel -> Ry rotation
            for i in range(self.wires):
                qml.RY(inputs[:, i], wires=i)
            # Ansatz layers
            for layer in range(self.layers):
                for i in range(self.wires):
                    qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
                # Entangling layer
                for i in range(0, self.wires - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, self.wires - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: input tensor of shape (batch, channels, height, width)
        :return: log probabilities of shape (batch, num_classes)
        """
        bsz = x.size(0)
        # Flatten to (batch, height, width)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, r:r + self.patch_size, c:c + self.patch_size]
                # Flatten patch to vector of length 4
                patch = patch.reshape(bsz, -1)
                # Evaluate circuit for each sample in batch
                out = self.circuit(patch, self.params)
                patches.append(out)
        # Concatenate all patch outputs
        features = torch.cat(patches, dim=1)  # shape: (batch, 4*196)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility of initial parameters.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

__all__ = ["QuanvolutionHybridClassifier"]
