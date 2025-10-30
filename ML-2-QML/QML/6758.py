"""Quantum kernel with parameter‑shared variational ansatz and trainable gamma.

The circuit encodes two feature vectors with opposite signs and measures the
probability of the all‑zero state.  A learnable parameter gamma scales the
rotation angles, enabling the kernel width to be optimized jointly with a
downstream model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(tq.QuantumModule):
    """Parameterized variational ansatz for data encoding.

    Parameters
    ----------
    n_wires : int
        Number of qubits used for encoding.
    gamma_init : float
        Initial value for the global scaling parameter.
    """
    def __init__(self, n_wires: int = 4, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y with opposite signs and reset between them."""
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] * self.gamma
            func_name_dict[info["func"]](q_device,
                                         wires=info["wires"],
                                         params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] * self.gamma
            func_name_dict[info["func"]](q_device,
                                         wires=info["wires"],
                                         params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap amplitude.

    It builds a :class:`KernalAnsatz` and uses it to compute the
    probability of measuring all qubits in state |0⟩.  The returned value
    is a differentiable similarity measure that can be used as a kernel.
    """
    def __init__(self, n_wires: int = 4, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires, gamma_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value for two batches of vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Probability of all‑zero outcome
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_wires: int = 4,
                  gamma_init: float = 1.0) -> np.ndarray:
    """Evaluate the Gram matrix between two collections of feature vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.  They are stacked into 2‑D tensors
        internally for efficient batch computation.
    n_wires : int, optional
        Number of qubits used in the ansatz.
    gamma_init : float, optional
        Initial value for the training parameter.

    Returns
    -------
    np.ndarray
        Gram matrix of shape ``(len(a), len(b))``.  The function is
        differentiable if the inputs are ``torch.Tensor`` with
        ``requires_grad=True``.
    """
    a_stack = torch.stack(a) if isinstance(a, (list, tuple)) else a
    b_stack = torch.stack(b) if isinstance(b, (list, tuple)) else b
    kernel = Kernel(n_wires, gamma_init)
    return kernel(a_stack, b_stack).detach().cpu().numpy()
