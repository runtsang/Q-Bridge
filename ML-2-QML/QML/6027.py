"""Hybrid quantum kernel module with a trainable feature map and overlap estimation."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["QuantumKernelMethod__gen444", "kernel_matrix"]

class QuantumKernelMethod__gen444(tq.QuantumModule):
    """
    Quantum kernel using a parameter‑shiftable feature map.

    The original seed used a fixed list of ry gates.  Here we introduce
    a multi‑layer ansatz with learnable rotation angles, allowing the
    kernel to adapt its mapping to the data distribution.  The
    kernel value is the absolute square of the overlap between the
    two encoded states, which can be differentiated with respect to
    the angles via the parameter‑shift rule provided by torchquantum.
    """

    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 init: str = "random") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Create a learnable parameter tensor for ry and rz rotations
        # shape: (n_layers, n_wires, 2) -> ry, rz per qubit per layer
        if init == "random":
            init_vals = torch.randn(n_layers, n_wires, 2)
        else:
            init_vals = torch.zeros(n_layers, n_wires, 2)
        self.params = torch.nn.Parameter(init_vals)
        # Build a static list of gate info for efficient execution
        self.gate_info = []
        for layer in range(n_layers):
            for qubit in range(n_wires):
                self.gate_info.append(
                    {"layer": layer, "qubit": qubit, "func": "ry", "wires": [qubit]}
                )
                self.gate_info.append(
                    {"layer": layer, "qubit": qubit, "func": "rz", "wires": [qubit]}
                )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode two classical vectors x and y into the same device and
        apply the ansatz.  The device is left in the state
        |ψ(x)⟩⨂|ψ(y)⟩; the kernel value is computed externally.
        """
        # Reset device to zero state
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.gate_info:
            layer = info["layer"]
            qubit = info["qubit"]
            idx = qubit  # use qubit index as feature index
            angle = self.params[layer, qubit, 0] * x[:, idx] if x.shape[1] > idx else 0.0
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=angle)
        # Encode y with negative angles
        for info in reversed(self.gate_info):
            layer = info["layer"]
            qubit = info["qubit"]
            idx = qubit
            angle = -self.params[layer, qubit, 1] * y[:, idx] if y.shape[1] > idx else 0.0
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=angle)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value |⟨ψ(x)|ψ(y)⟩|^2.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        # Overlap between first two states
        overlap = torch.abs(self.q_device.states.view(-1)[0])
        return overlap ** 2

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the quantum kernel Gram matrix between two lists of tensors.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        List of tensors, each of shape (features,)
    b : Sequence[torch.Tensor]
        List of tensors, each of shape (features,)

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b))
    """
    kernel = QuantumKernelMethod__gen444()
    a_batch = torch.stack(a)
    b_batch = torch.stack(b)
    with torch.no_grad():
        mat = torch.zeros((len(a), len(b)))
        for i, x in enumerate(a_batch):
            for j, y in enumerate(b_batch):
                mat[i, j] = kernel.kernel_value(x, y)
        return mat.cpu().numpy()
