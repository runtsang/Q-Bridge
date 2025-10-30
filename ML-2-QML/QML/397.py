"""ConvEnhanced: quantum convolution filter using PennyLane.

This module implements a variational quantum filter that can be used
as a drop‑in replacement for the original Conv class.  The filter
applies a parameterised rotation to each qubit representing a pixel
of the kernel, entangles the qubits, and returns the mean probability
of measuring |1> across all qubits.

"""

import pennylane as qml
import torch
from torch import nn
import numpy as np

# Device for a 2x2 kernel (4 qubits)
dev = qml.device("default.qubit", wires=4)

def _variational_layer(params: torch.Tensor):
    """Apply a two‑layer rotation network."""
    for i in range(4):
        qml.RX(params[i], wires=i)
        qml.RY(params[i + 4], wires=i)
    # Entangling layer
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def _qnode(pixels: torch.Tensor, params: torch.Tensor):
    _variational_layer(params)
    # Return the mean probability of measuring |1> on each qubit
    probs = [qml.probs(wires=i)[0] for i in range(4)]
    return sum(probs) / 4

__all__ = ["ConvEnhanced"]

class ConvEnhanced(nn.Module):
    """Quantum convolution filter implemented with PennyLane."""
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        if self.n_qubits!= 4:
            raise ValueError("Current implementation supports only 2x2 kernels.")
        # Learnable rotation parameters (2 per qubit)
        self.params = nn.Parameter(torch.randn(2 * self.n_qubits))
        self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to the first patch of each sample."""
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            patch = x[i, 0, :self.kernel_size, :self.kernel_size]
            # Flatten and normalize to [0,1]
            pixels = patch.view(-1) / 255.0
            out = _qnode(pixels, self.params)
            outputs.append(out)
        return torch.stack(outputs)
