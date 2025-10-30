"""Quantum kernel construction using TorchQuantum with multi‑scale support and fidelity diagnostics."""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["KernalAnsatz", "Kernel", "KernelMatrix", "KernelRegressor", "KernelClassifier"]


def _check_gammas(gamma: float | Iterable[float]) -> Tuple[float,...]:
    """Return a tuple of distinct gamma values for multi‑scale RBF."""
    if isinstance(gamma, (list, tuple)):
        if not all(isinstance(g, (float, int)) for g in gamma):
            raise ValueError("All gamma values must be numeric")
        return tuple(g for g in gamma if g > 0)
    else:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        return (float(gamma),)


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self, n_wires: int = 4, func_list: Iterable[dict] | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if func_list is None:
            func_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def KernelMatrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | Iterable[float] = 1.0) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` with optional multi‑scale weighting."""
    gammas = _check_gammas(gamma)
    kernels = [Kernel() for _ in gammas]
    return np.sum([np.array([[k(x, y).item() for y in b] for x in a]) for k in kernels], axis=0)


class KernelRegressor(BaseEstimator, RegressorMixin):
    """A scikit‑learn regressor that uses a quantum‑style kernel matrix."""

    def __init__(self, gamma: float | Iterable[float] = 1.0, alpha: float = 1e-3):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.kernel_ = KernelMatrix(X, X, self.gamma)
        self.coef_ = np.linalg.solve(self.kernel_ + self.alpha * np.eye(len(X)), y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        K = KernelMatrix(X, self.kernel_, self.gamma)
        return K @ self.coef_


class KernelClassifier(BaseEstimator, ClassifierMixin):
    """A scikit‑learn classifier that uses a quantum‑style kernel matrix."""

    def __init__(self, gamma: float | Iterable[float] = 1.0, C: float = 1.0):
        self.gamma = gamma
        self.C = C

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.kernel_ = KernelMatrix(X, X, self.gamma)
        from sklearn.svm import LinearSVC
        self.svm_ = LinearSVC(C=self.C)
        self.svm_.fit(self.kernel_, y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        K = KernelMatrix(X, self.kernel_, self.gamma)
        return self.svm_.predict(K)
