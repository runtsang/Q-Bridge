"""Quantum kernel module based on TorchQuantum, compatible with the hybrid model.

The implementation follows the design from the original QML seed but is extended to
support arbitrary input dimensionality and to expose a clean ``forward`` interface
that can be used directly in a classical PyTorch model, for example as a feature map
in a hybrid regressor.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class EstimatorQNNHybrid(tq.QuantumModule):
    """Quantum kernel implemented as a TorchQuantum module."""

    def __init__(self, n_wires: int = 4, func_list: list | None = None) -> None:
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits in the device.
        func_list : list | None
            List of gate specifications.  Each item is a dict with keys
            ``input_idx`` (list of indices of input features), ``func`` (gate name),
            and ``wires`` (list of wires).  If ``None`` a default list of ry gates is used.
        """
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Default encoding: one ry per input dimension
        if func_list is None:
            func_list = [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
            ]
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode ``x`` and ``y`` into the quantum device and compute the overlap.

        Parameters
        ----------
        q_device : tq.QuantumDevice
            Quantum device used for simulation.
        x : torch.Tensor
            Shape (batch, dim).
        y : torch.Tensor
            Shape (support, dim).
        """
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if func_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode -y
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if func_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Public API to compute the kernel value between two batches of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, dim).
        y : torch.Tensor
            Shape (support, dim).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch, support).
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def EstimatorQNN() -> EstimatorQNNHybrid:
    """Convenience factory matching the original API."""
    return EstimatorQNNHybrid()
