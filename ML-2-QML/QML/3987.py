"""Hybrid quantum kernel module.

The quantum implementation follows the TorchQuantum API and mirrors
the classical RBF ansatz.  It also exposes a QCNN‑style quantum
convolutional block that can be chained with classical layers.
"""

from __future__ import annotations

from typing import Callable, Sequence, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# 1.  Quantum RBF ansatz
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel that encodes two vectors via inverse Ry rotations."""
    def __init__(self, n_wires: int, input_dim: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.input_dim = input_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset the device for each batch
        self.q_device.reset_states(x.shape[0])

        # Encode first vector with positive rotations
        for i in range(self.input_dim):
            func_name_dict["ry"](self.q_device, wires=i, params=x[:, i])

        # Encode second vector with negative rotations (inverse)
        for i in range(self.input_dim):
            func_name_dict["ry"](self.q_device, wires=i, params=-y[:, i])

    def overlap(self) -> torch.Tensor:
        """Return the absolute overlap of the first qubit state."""
        return torch.abs(self.q_device.states.view(-1)[0])


class Kernel(tq.QuantumModule):
    """Wrapper that evaluates the quantum RBF kernel."""
    def __init__(self, n_wires: int = 4, input_dim: int = 4) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires, input_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(x, y)
        return self.ansatz.overlap()


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_wires: int = 4,
                  input_dim: int = 4) -> np.ndarray:
    """Compute Gram matrix with the quantum kernel."""
    kernel = Kernel(n_wires, input_dim)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 2.  Hybrid quantum‑classical kernel
# --------------------------------------------------------------------------- #

class HybridKernel(tq.QuantumModule):
    """
    Weighted sum of a classical RBF kernel and a quantum kernel.
    The classical part is evaluated on the CPU, the quantum part
    on the device.  ``weight`` ∈ [0,1] controls the blending.
    """
    def __init__(self,
                 n_wires: int,
                 input_dim: int,
                 gamma: float = 1.0,
                 weight: float = 0.5) -> None:
        super().__init__()
        self.quantum = KernalAnsatz(n_wires, input_dim)
        self.gamma = gamma
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Quantum overlap
        self.quantum(x, y)
        Kq = self.quantum.overlap()

        # Classical RBF on CPU
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        Kc = torch.exp(-self.gamma * torch.sum(diff * diff, dim=2))

        return (1 - self.weight) * Kc + self.weight * Kq


def hybrid_kernel_matrix(a: Sequence[torch.Tensor],
                         b: Sequence[torch.Tensor],
                         n_wires: int,
                         input_dim: int,
                         gamma: float = 1.0,
                         weight: float = 0.5) -> np.ndarray:
    kernel = HybridKernel(n_wires, input_dim, gamma, weight)
    return kernel(a, b).detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# 3.  QCNN‑style quantum convolution / pooling layers
# --------------------------------------------------------------------------- #

class QCNNQuantumBlock(tq.QuantumModule):
    """
    A compact 2‑qubit convolution block that mirrors the classical
    layers in ``QCNNModel``.  The block can be chained to form a
    depth‑wise QCNN.
    """
    def __init__(self, wires: Sequence[int]) -> None:
        super().__init__()
        self.wires = wires
        self.q_device = tq.QuantumDevice(n_wires=max(wires)+1)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> None:
        # Simple template: RY – CNOT – RY – CNOT
        self.q_device.reset_states(x.shape[0])
        for w in self.wires:
            self.q_device.apply(tq.ry, wires=[w], params=x[:, w])

        # Coupling via CNOTs
        if len(self.wires) >= 2:
            self.q_device.apply(tq.cnot, wires=[self.wires[0], self.wires[1]])

    def output_state(self) -> torch.Tensor:
        return self.q_device.states.view(-1)[0]


class QCNNQuantumModel(tq.QuantumModule):
    """
    A light‑weight QCNN that stacks multiple ``QCNNQuantumBlock`` s
    and uses a trainable Pauli observable for classification.
    """
    def __init__(self, num_layers: int = 3, n_wires: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([QCNNQuantumBlock(range(i, i+2))
                                     for i in range(0, n_wires, 2)][:num_layers])
        # Single‑qubit Z observable on last wire
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_wires-1), 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            block(x)
        # Compute expectation value
        return torch.abs(self.blocks[-1].output_state())


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernel",
    "hybrid_kernel_matrix",
    "QCNNQuantumBlock",
    "QCNNQuantumModel",
]
