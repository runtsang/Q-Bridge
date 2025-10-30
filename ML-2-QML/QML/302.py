"""Quanvolutional filter inspired by ``quanvolution.py`` in the raw dataset.
This version uses a variational quantum circuit per 2×2 patch and
supports back‑propagation via PennyLane's autograd support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Optional

class QuanvolutionFilter(nn.Module):
    """Apply a trainable variational quantum kernel to 2×2 image patches.

    The circuit consists of a single layer of rotations followed by a
    CNOT ladder.  The parameters of the rotations are trainable and
    updated through the quantum gradient.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # device with autograd support
        self.device = qml.device("default.qubit.autograd", wires=self.n_wires)
        # number of parameters: n_wires * n_layers
        self.n_params = self.n_wires * self.n_layers
        # initialize parameters
        self.params = nn.Parameter(torch.randn(self.n_params))
        # define the quantum node
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # encode the 4 pixel values into rotation angles
            for i in range(self.n_wires):
                qml.RY(inputs[i], wires=i)
            # variational layer
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_wires):
                    qml.RZ(params[idx], wires=i)
                    idx += 1
                # entangling CNOT ladder
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # measurement: expectation of PauliZ on all wires
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), \
                   qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the variational quantum filter to each 2×2 patch."""
        bsz = x.size(0)
        # reshape to (batch, 28, 28)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # extract 2x2 patch and flatten
                patch = x[:, r:r+2, c:c+2].reshape(bsz, -1)
                # run the circuit
                out = self.circuit(patch, self.params)
                # out is a tuple of 4 expectation values -> shape (batch, 4)
                out = torch.stack(out, dim=1)
                patches.append(out)
        # concatenate all patches along feature dimension
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_wires=n_wires, n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
