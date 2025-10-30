"""Hybrid quantum kernel combining a TorchQuantum ansatz with a linear feature map."""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence


class HybridQuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that encodes data via a parameterized ansatz and
    augments the amplitude overlap with a classical linear mapping.
    The kernel value between ``x`` and ``y`` is:
        k(x, y) = |⟨0|U†(x)U(y)|0⟩| + ⟨f(x), f(y)⟩
    where ``U`` is the ansatz and ``f`` is a learnable linear layer
    applied to the classical inputs.
    """
    def __init__(self, n_wires: int = 4, n_features: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple RY‑only ansatz: one RY per input dimension
        self.ansatz = tq.QuantumModule()
        for i in range(n_features):
            self.ansatz.add_layer(
                tq.RY, wires=[i], params="theta{}".format(i)
            )
        # Classical linear mapping
        self.linear = torch.nn.Linear(n_features, 1, bias=True)
        self.activation = torch.nn.Tanh()

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode x
        x_params = {f"theta{i}": x[:, i] for i in range(x.shape[1])}
        self.ansatz(self.q_device, **x_params)
        amp_x = torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

        # Encode y
        y_params = {f"theta{i}": y[:, i] for i in range(y.shape[1])}
        self.ansatz(self.q_device, **y_params)
        amp_y = torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

        # Overlap term
        overlap = torch.abs(torch.dot(amp_x, amp_y))

        # Classical linear mapping
        fx = self.activation(self.linear(x))
        fy = self.activation(self.linear(y))
        fc_dot = torch.mm(fx.t(), fy).squeeze()

        return overlap + fc_dot


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute Gram matrix using the hybrid quantum kernel.
    """
    kernel = HybridQuantumKernel(n_wires=4, n_features=a[0].shape[0])
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridQuantumKernel", "kernel_matrix"]
