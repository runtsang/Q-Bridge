"""Hybrid quantum kernel combining a quantum convolution layer with a variational Ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class ConvAnsatz(tq.QuantumModule):
    """Quantum convolution encoder that maps classical data onto qubit rotations.

    Parameters
    ----------
    conv_size : int
        Size of the square filter; the number of qubits is ``conv_size**2``.
    threshold : float
        Threshold used to decide whether a rotation is applied.
    """

    def __init__(self, conv_size: int, threshold: float) -> None:
        super().__init__()
        self.conv_size = conv_size
        self.threshold = threshold
        self.gates = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(conv_size**2)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Encode ``x`` using Ry rotations conditioned on a threshold."""
        for info in self.gates:
            params = x[:, info["input_idx"]]
            # Convert to pi or 0 based on threshold.
            params = torch.where(params > self.threshold, torch.pi, torch.zeros_like(params))
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class KernelAnsatz(tq.QuantumModule):
    """Variational Ansatz used to compute the overlap between two data points."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply the Ansatz to encode ``x`` and ``y`` on the same device."""
        # Encode x
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode y with reversed order and negative parameters
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class HybridKernel(tq.QuantumModule):
    """Quantum kernel that first applies a quantum convolution then a variational Ansatz.

    Parameters
    ----------
    n_wires : int
        Number of qubits used by the device. Must be at least ``conv_size**2``.
    conv_size : int
        Size of the square filter for the quantum convolution.
    conv_threshold : float
        Threshold for the convolutional rotation.
    """

    def __init__(
        self,
        n_wires: int = 4,
        conv_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

        # Quantum convolution encoder
        self.conv_ansatz = ConvAnsatz(conv_size, conv_threshold)

        # Variational Ansatz – a simple layered circuit.
        func_list = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]
        self.kernel_ansatz = KernelAnsatz(func_list)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value for two 1‑D vectors.

        The vectors are first encoded via the quantum convolution layer,
        then passed through the variational Ansatz to compute the overlap.
        """
        # Ensure input shape: (1, conv_size**2)
        size = self.kernel_ansatz.func_list[0]["input_idx"][0] + 1
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        self.q_device.reset_states(1)
        self.conv_ansatz(self.q_device, x)
        self.kernel_ansatz(self.q_device, x, y)

        # Return the absolute value of the first amplitude as the kernel value.
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix for two collections of vectors."""
        return np.array(
            [[self(x, y).item() for y in b] for x in a]
        )


__all__ = ["HybridKernel"]
