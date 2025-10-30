"""Quantum kernel construction using a variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["HybridKernel", "kernel_matrix"]


class HybridKernel(tq.QuantumModule):
    """
    A quantum kernel that implements a variational feature map.

    The circuit encodes two input vectors ``x`` and ``y`` into the same
    device, then applies a parameterised ansatz that depends on both
    inputs.  The kernel value is the absolute overlap of the resulting
    state with the computational basis state ``|0…0⟩``.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Trainable parameters for the variational ansatz
        self.theta = torch.nn.Parameter(torch.randn(self.n_layers * self.n_wires))

        # Build the variational ansatz as a list of gate dictionaries
        self.ansatz = []
        idx = 0
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                self.ansatz.append(
                    {
                        "func": "ry",
                        "wires": [wire],
                        "params": [self.theta[idx]],
                    }
                )
                idx += 1
            for wire in range(self.n_wires - 1):
                self.ansatz.append(
                    {
                        "func": "cx",
                        "wires": [wire, wire + 1],
                        "params": None,
                    }
                )

    def encode(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode classical data into the quantum device by applying a
        data‑driven rotation on each wire.  The two vectors are encoded
        sequentially with opposite signs to create an interference pattern.
        """
        q_device.reset_states(x.shape[0])
        for idx, val in enumerate(x):
            func_name_dict["ry"](q_device, wires=[idx], params=val)
        for idx, val in enumerate(y):
            func_name_dict["ry"](q_device, wires=[idx], params=-val)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the variational quantum kernel.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (n_wires,).

        Returns
        -------
        torch.Tensor
            The kernel value as a scalar tensor.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.encode(self.q_device, x, y)

        # Apply the variational ansatz
        for gate in self.ansatz:
            if gate["params"] is None:
                func_name_dict[gate["func"]](self.q_device, wires=gate["wires"])
            else:
                func_name_dict[gate["func"]](self.q_device, wires=gate["wires"], params=gate["params"])

        # Return the absolute overlap with |0…0⟩
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix for a list of samples ``a`` against ``b`` using
    the variational quantum kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors of equal dimensionality.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape (len(a), len(b)).
    """
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
