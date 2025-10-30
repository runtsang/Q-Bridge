"""Quantum hybrid kernel that encodes convolutional features into a TorchQuantum circuit.

The QML implementation mirrors the classical `HybridKernelMethod` but uses a
quantum circuit to evaluate the similarity.  Convolutional feature extraction
is performed classically and the resulting scalar is used as a rotation angle
for each qubit, thereby fusing classical and quantum information in the
kernel evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Programmable quantum feature map that applies a list of gates.

    The list is defined by ``func_list`` where each entry specifies the gate
    name, the wires it acts on, and the index of the input data that supplies
    the rotation angle.
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(1)
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two feature states."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the quantum Gram matrix for two datasets."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Classical convolutional filter used as a feature extractor for the quantum kernel
# --------------------------------------------------------------------------- #
class ConvFilter:
    """Simple 2‑D convolutional filter that returns a scalar feature."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Initialise a random weight matrix to mimic a quantum-inspired filter
        self.weights = np.random.randn(kernel_size, kernel_size)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to data and return a scalar."""
        conv = np.tensordot(data, self.weights, axes=([0, 1], [0, 1]))
        activated = 1.0 / (1.0 + np.exp(-(conv - self.threshold)))
        return float(activated)


# --------------------------------------------------------------------------- #
# Hybrid quantum kernel that incorporates convolutional features into gate angles
# --------------------------------------------------------------------------- #
class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel that first encodes a convolutional feature as a rotation angle.

    The convolutional feature is computed classically and then used to parameterise
    Ry gates on each qubit.  The kernel value is the absolute overlap of the
    resulting quantum states.
    """
    def __init__(self, n_wires: int = 4, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.conv = ConvFilter(kernel_size, threshold)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode convolutional features into the quantum circuit and return similarity."""
        # Compute convolutional scalars
        conv_x = self.conv.run(x.numpy())
        conv_y = self.conv.run(y.numpy())

        # Encode conv_x into the first pass of the ansatz
        x_tensor = torch.full((1, self.n_wires), conv_x, dtype=torch.float32)
        y_tensor = torch.full((1, self.n_wires), conv_y, dtype=torch.float32)

        self.ansatz(self.q_device, x_tensor, y_tensor)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the hybrid quantum Gram matrix for two datasets."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "ConvFilter",
    "HybridKernelMethod",
]
