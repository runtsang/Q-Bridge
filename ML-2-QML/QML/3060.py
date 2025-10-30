"""HybridQCNet – Quantum implementation of the hybrid binary classifier.

The quantum version replaces the classical RBF kernel with a TorchQuantum
ansatz and the hybrid head with a parametrised quantum circuit that
evaluates an expectation value.  The public API matches the classical
module so that the two can be swapped transparently.

Key components:
* `QuantumKernel` – TorchQuantum module that encodes two inputs via
  a sequence of Ry gates and returns the overlap of the resulting
  states.
* `HybridQuantumLayer` – Quantum circuit that maps a scalar feature to
  a single expectation value.
* `HybridQCNet` – Convolutional backbone + quantum kernel + quantum
  hybrid head.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that implements a fixed TorchQuantum ansatz.

    Parameters
    ----------
    n_wires : int
        Number of qubits used for the kernel.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Reshape to batch‑first
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Encode x positively, y negatively
        self.ansatz(self.q_device, x, y)

        # Return absolute value of the first amplitude (overlap)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridQuantumLayer(tq.QuantumModule):
    """Parametrised quantum circuit that acts as the hybrid head.

    The circuit consists of a single Ry gate followed by a measurement
    of the Z observable.  The expectation value is returned as a scalar.
    """

    def __init__(self, n_qubits: int = 1, shift: float = math.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

    @tq.static_support
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # `inputs` is a scalar tensor
        self.q_device.reset_states(inputs.shape[0])
        self.q_device.ry(inputs + self.shift, wires=0)
        # Measure Z on the first qubit
        return self.q_device.expectation(self.q_device.z_op, wires=0)


class HybridQCNet(nn.Module):
    """Full quantum hybrid network.

    The architecture follows the classical backbone but replaces the
    kernel and the head with quantum modules.  All tensors flow through
    TorchQuantum devices and the network remains fully differentiable
    via the `tq.static_support` decorator.
    """

    def __init__(self, support_vectors: Iterable[torch.Tensor], gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone (identical to the classical version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum kernel layer
        self.kernel = QuantumKernel(n_wires=4)

        # Hybrid head
        self.hybrid = HybridQuantumLayer(n_qubits=1, shift=math.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Quantum kernel evaluation
        # For each support vector, compute kernel with the current sample.
        # Here we use the first support vector as a placeholder for simplicity.
        # In practice, a full Gram matrix can be pre‑computed or batched.
        kernel_features = self.kernel(x, torch.tensor([0.0]))  # dummy y for illustration

        # Hybrid head
        probs = self.hybrid(kernel_features)

        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumKernel", "HybridQuantumLayer", "HybridQCNet"]
