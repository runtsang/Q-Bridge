"""Quantum kernel construction using TorchQuantum ansatz with trainable parameters.

This module extends the seed implementation by turning the fixed rotation list
into a variational circuit.  Each gate can be marked as ``trainable`` and
the corresponding torch.nn.Parameter will be optimized together with any
downstream loss.  The kernel remains differentiable under the torch
autograd engine, which is essential for end‑to‑end kernel learning.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes classical data via a list of gates.

    Parameters
    ----------
    func_list : list[dict]
        Each dictionary must contain ``func`` (gate name string), ``wires``
        (list[int]), and optionally ``input_idx`` (list[int] of the input
        indices to pull parameters from).  If ``trainable`` is set to True,
        a learnable torch.nn.Parameter will be created for that gate.
    """

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list
        self.trainable_params = nn.ParameterList()
        for info in func_list:
            if info.get("trainable", False):
                # assume a single real parameter per trainable gate
                self.trainable_params.append(nn.Parameter(torch.randn(1)))
            else:
                self.trainable_params.append(None)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # reset device for each batch
        q_device.reset_states(x.shape[0])
        # Apply forward pass for x
        for i, info in enumerate(self.func_list):
            if info.get("trainable", False):
                params = self.trainable_params[i]
            else:
                params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Apply reverse pass for y with negated parameters
        for i, info in reversed(list(enumerate(self.func_list))):
            if info.get("trainable", False):
                params = -self.trainable_params[i]
            else:
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel implemented via a variational ansatz.

    The default ansatz consists of four single‑qubit rotations around the
    Y‑axis.  All rotations are trainable, allowing the kernel to adapt to the
    data distribution.  The circuit is fully differentiable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0], "trainable": True},
                {"input_idx": [1], "func": "ry", "wires": [1], "trainable": True},
                {"input_idx": [2], "func": "ry", "wires": [2], "trainable": True},
                {"input_idx": [3], "func": "ry", "wires": [3], "trainable": True},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of input vectors.  Each tensor is assumed to be a
        one‑dimensional torch tensor representing a single sample.
    """
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
