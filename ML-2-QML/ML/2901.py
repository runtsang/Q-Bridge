"""Hybrid kernel module combining classical RBF and variational quantum kernel.

The module introduces:
* `HybridKernel` – a weighted sum of a trainable RBF kernel and a
  variational quantum kernel built with TorchQuantum.  The weight
  ``alpha`` controls the classical/quantum trade‑off.
* `HybridFCL` – a fully‑connected layer that can operate either in a
  purely classical mode (torch.nn.Linear) or in a quantum mode
  (parameterized Ry gates).  The same ``run`` interface is provided
  for both variants.

Both classes expose a ``forward``/``run`` method that returns a scalar
tensor or NumPy array, making them drop‑in replacements for the
original seed classes while offering richer expressivity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Iterable

# --------------------------------------------------------------------------- #
# Classical RBF kernel with trainable γ
# --------------------------------------------------------------------------- #
class RBFAnsatz(nn.Module):
    """Trainable radial‑basis‑function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # γ is a learnable parameter to allow the kernel width to adapt
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-γ‖x−y‖²)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Variational quantum kernel
# --------------------------------------------------------------------------- #
class QuantumAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes data via Ry rotations and
    performs a reverse encoding with negative angles.

    The circuit is a simple layer of Ry gates; it can be expanded
    with additional entangling gates for future experiments.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum kernel for a single pair of vectors."""
        self.q_device.reset_states(x.shape[0])

        # Forward encoding
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Reverse encoding with negative angles
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Return the overlap between the two states
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# Hybrid kernel that blends the two kernels
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Weighted sum of a classical RBF kernel and a variational quantum kernel.

    Parameters
    ----------
    gamma : float
        Initial width of the classical RBF kernel.
    alpha : float
        Weight of the classical component (0 ≤ α ≤ 1).
    n_wires : int
        Number of qubits used by the quantum ansatz.
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 0.5, n_wires: int = 4) -> None:
        super().__init__()
        self.alpha = alpha
        self.classical = RBFAnsatz(gamma)
        self.quantum = QuantumAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        cls = self.classical(x, y).squeeze()
        qk = self.quantum(x, y)
        return self.alpha * cls + (1.0 - self.alpha) * qk

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, alpha: float = 0.5, n_wires: int = 4) -> np.ndarray:
    """Compute Gram matrix for two collections of vectors."""
    kernel = HybridKernel(gamma, alpha, n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid fully‑connected layer
# --------------------------------------------------------------------------- #
class HybridFCL(nn.Module):
    """A fully‑connected layer that can operate classically or quantumly.

    The ``run`` method accepts an iterable of parameters and returns a
    NumPy array of the layer output.  In classical mode the layer
    consists of a single linear transform followed by a tanh non‑linearity.
    In quantum mode a single‑qubit Ry circuit is executed on a simulated
    backend.
    """

    def __init__(self, n_features: int = 1, use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if self.use_quantum:
            # Quantum circuit configuration
            self.n_qubits = 1
            self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        else:
            self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        if self.use_quantum:
            q_device = tq.QuantumDevice(n_wires=1)
            q_device.reset_states(1)
            for theta in thetas:
                tq.ry(q_device, wires=[0], params=theta)
            prob = torch.abs(q_device.states.view(-1)[0]) ** 2
            return prob.detach().numpy()
        else:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

__all__ = ["HybridKernel", "kernel_matrix", "HybridFCL"]
