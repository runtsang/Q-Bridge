"""Hybrid convolutional kernel with quantum RBF similarity.

This module implements a quantum version of the hybrid convolutional filter.
It encodes a 2‑D patch into a quantum device using a simple rotation ansatz
and evaluates the overlap with a fixed kernel weight vector.  The design
mirrors the classical implementation while leveraging quantum resources
for the similarity computation.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class HybridConvKernel(tq.QuantumModule):
    """
    Quantum hybrid convolutional kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the square convolution kernel.
    threshold : float
        Threshold used to decide whether to rotate a qubit.
    shots : int
        Number of shots for the quantum device (used only when a backend
        requiring sampling is employed).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 100,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots

        self.n_qubits = kernel_size ** 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.ansatz = self._build_ansatz()

        # Random kernel weights that act as the second input to the kernel
        self.kernel_weights = torch.randn(self.n_qubits)

    def _build_ansatz(self):
        """Return a list of gate specifications for the ansatz."""
        return [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_qubits)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode two inputs into the quantum device and compute the overlap.

        Parameters
        ----------
        q_device : tq.QuantumDevice
            The quantum device to run the circuit on.
        x : torch.Tensor
            First input vector of shape (n_qubits,).
        y : torch.Tensor
            Second input vector of shape (n_qubits,).
        """
        q_device.reset_states(x.shape[0])
        # Encode first input
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode second input with negative parameters
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value for two input vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        # Return the absolute amplitude of the |0...0> state
        return torch.abs(self.q_device.states.view(-1)[0])

    def run(self, data: np.ndarray) -> float:
        """
        Compute the kernel between a 2‑D patch and the internal kernel weights.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Quantum kernel value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(-1)
        kernel_val = self.kernel_value(tensor, self.kernel_weights)
        return kernel_val.item()


__all__ = ["HybridConvKernel"]
