"""Hybrid classical‑quantum kernel module.

The :class:`QuantumKernelMethod` class exposes a composite kernel that is
the weighted sum of a classical RBF kernel and a quantum kernel implemented
with TorchQuantum.  It also offers a simple grid‑search tuner for the
bandwidth ``gamma`` and the quantum weight ``q_weight``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# 1.  Classical RBF kernel
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    gamma : float, default=1.0
        Bandwidth parameter.
    normalize : bool, default=True
        Normalise the kernel to unit diagonal.
    """
    def __init__(self, gamma: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (N, M, D)
        sq_norm = torch.sum(diff * diff, dim=-1)        # (N, M)
        k = torch.exp(-self.gamma * sq_norm)
        if self.normalize:
            k = k / torch.sqrt(k.diagonal(dim1=-2, dim2=-1).unsqueeze(1))
        return k

# --------------------------------------------------------------------------- #
# 2.  Quantum kernel (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumAnsatz(tq.QuantumModule):
    """Simple ansatz that encodes two vectors onto the same qubits."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        self.ansatz(self.q_device, x, y)
        # The first element of the state is the overlap
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# 3.  Hybrid kernel
# --------------------------------------------------------------------------- #
class QuantumKernelMethod(nn.Module):
    """Weighted product of a quantum kernel and an RBF kernel.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF bandwidth.
    q_weight : float, default=0.5
        Weight of the quantum kernel (∈ [0, 1]).
    n_wires : int, default=4
        Number of qubits in the quantum circuit.
    normalize : bool, default=True
        Normalise the RBF component.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        q_weight: float = 0.5,
        n_wires: int = 4,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma=gamma, normalize=normalize)
        self.qk = QuantumKernel(n_wires=n_wires)
        self.q_weight = torch.tensor(q_weight, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf = self.rbf(x, y)
        qk = self.qk(x, y)
        return (self.q_weight * qk) + ((1.0 - self.q_weight) * rbf)

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two batches of tensors."""
        X = torch.stack(X)
        Y = torch.stack(Y)
        return self.forward(X, Y).detach().cpu().numpy()

    @staticmethod
    def tune(
        X: Sequence[torch.Tensor],
        Y: Sequence[torch.Tensor],
        gamma_grid: Tuple[float,...] = (0.1, 1.0, 10.0),
        q_weight_grid: Tuple[float,...] = (0.0, 0.5, 1.0),
    ) -> Tuple[float, float]:
        """Return (best_gamma, best_q_weight) that maximises the trace of the
        kernel matrix on the training data (a simple heuristic)."""
        best_score = -np.inf
        best_params = (gamma_grid[0], q_weight_grid[0])
        for gamma in gamma_grid:
            for q_weight in q_weight_grid:
                model = QuantumKernelMethod(gamma=gamma, q_weight=q_weight)
                K = model.kernel_matrix(X, Y)
                score = np.trace(K)
                if score > best_score:
                    best_score = score
                    best_params = (gamma, q_weight)
        return best_params

__all__ = ["QuantumKernelMethod", "RBFKernel", "QuantumKernel"]
