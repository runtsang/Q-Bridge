"""Quantum hybrid kernel implementation using TorchQuantum and classical RBF."""

from __future__ import annotations

from typing import Sequence, Iterable, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn

__all__ = ["HybridKernel"]


class ClassicalRBF(nn.Module):
    """Pure classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))


class QuantumAnsatz(tq.QuantumModule):
    """Quantum feature map ansatz."""
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


class QuantumFeatureMap(tq.QuantumModule):
    """Wrapper that holds a quantum device and ansatz."""
    def __init__(self,
                 n_wires: int = 4,
                 func_list: Optional[list] = None) -> None:
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
        self.ansatz = QuantumAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0:1])


class HybridKernel(tq.QuantumModule):
    """Hybrid kernel combining classical RBF and quantum kernel with weight alpha."""
    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 1.0,
                 n_wires: int = 4,
                 func_list: Optional[list] = None,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.classical = ClassicalRBF(gamma=gamma)
        self.quantum = QuantumFeatureMap(n_wires=n_wires, func_list=func_list)
        self.device = device or torch.device("cpu")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        k_class = self.classical(x, y)
        k_quantum = self.quantum(x, y)
        return self.alpha * k_class + (1 - self.alpha) * k_quantum

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        a = torch.stack([t.to(self.device) for t in a])
        b = torch.stack([t.to(self.device) for t in b])
        return self.forward(a, b).detach().cpu().numpy()

    def fit(self, X: Sequence[np.ndarray], y: Sequence[int]) -> None:
        """Placeholder fit method."""
        pass

    def predict(self,
                X: Sequence[np.ndarray],
                X_train: Sequence[np.ndarray],
                y_train: Sequence[int]) -> np.ndarray:
        K = self.kernel_matrix(X_train, X_train)
        K_inv = np.linalg.pinv(K)
        y_train = np.array(y_train)
        predictions = []
        for x in X:
            k_x = self.kernel_matrix([x], X_train)[0]
            pred = k_x @ K_inv @ y_train
            predictions.append(pred)
        return np.array(predictions)
