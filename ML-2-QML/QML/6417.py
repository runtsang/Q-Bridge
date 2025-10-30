"""Quantum kernel implementation using TorchQuantum with trainable parameters.

The kernel is defined by a fixed ansatz of single‑qubit rotations whose
rotation angles are learnable.  The ansatz is applied to encode a data
vector ``x`` and then re‑encoded with the negative of another vector
``y``.  The overlap of the resulting states is used as the kernel value.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Parameterised ansatz that encodes two data vectors.

    Parameters
    ----------
    gate_list : list[dict]
        Each entry specifies a gate to apply with the following keys:
        * ``func`` – gate name understood by TorchQuantum (e.g. ``ry``).
        * ``wires`` – list of wires to act on.
        * ``param_idx`` – index of the feature to map to the rotation angle.
        * ``trainable`` – whether the corresponding angle is a learnable
          parameter (default: ``True``).
    """

    def __init__(self, gate_list):
        super().__init__()
        self.gate_list = gate_list
        self.params = nn.ParameterList()
        for g in self.gate_list:
            if g.get("trainable", True):
                self.params.append(nn.Parameter(torch.rand(1)))
            else:
                self.params.append(nn.Parameter(torch.tensor(0.0), requires_grad=False))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode ``x`` and ``y`` on the device and compute the overlap.
        """
        q_device.reset_states(x.shape[0])

        # Encode x
        for i, g in enumerate(self.gate_list):
            params = x[:, g["param_idx"]]
            if g.get("trainable", True):
                params = params * self.params[i]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)

        # Encode y with a negative sign
        for i, g in reversed(list(enumerate(self.gate_list))):
            params = -y[:, g["param_idx"]]
            if g.get("trainable", True):
                params = params * self.params[i]
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that uses the parameterised ``KernalAnsatz``."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"param_idx": 0, "func": "ry", "wires": [0], "trainable": True},
                {"param_idx": 1, "func": "ry", "wires": [1], "trainable": True},
                {"param_idx": 2, "func": "ry", "wires": [2], "trainable": True},
                {"param_idx": 3, "func": "ry", "wires": [3], "trainable": True},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the absolute overlap between the two encoded states.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
