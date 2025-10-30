"""Variational quantum kernel with trainable parameters and hybrid training loop."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from torch.optim import Adam
from torchquantum.functional import func_name_dict

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(tq.QuantumModule):
    """Variational quantum kernel with trainable ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Trainable parameters for each layer and wire
        self.params = nn.Parameter(torch.randn(depth, n_wires))
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """Return a list of gate specifications for the ansatz."""
        func_list = []
        for d in range(self.depth):
            for w in range(self.n_wires):
                func_list.append({"input_idx": [w], "func": "ry", "wires": [w]})
        return func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y into the quantum device and apply the ansatz."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for idx, info in enumerate(self.ansatz):
            param = self.params[info["input_idx"][0], info["wires"][0]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=param * x[:, info["input_idx"][0]])
        # Encode y with negative parameters
        for idx, info in reversed(list(enumerate(self.ansatz))):
            param = self.params[info["input_idx"][0], info["wires"][0]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=-param * y[:, info["input_idx"][0]])

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a pair of inputs."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        a_t = torch.stack(a)
        b_t = torch.stack(b)
        K = torch.stack([self.kernel_value(a_t[i:i+1], b_t) for i in range(len(a_t))])
        return K.detach().cpu().numpy()

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """Optimize the ansatz parameters by minimizing the negative log marginal likelihood."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        optimizer = Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            K = torch.tensor(self.kernel_matrix([torch.tensor(v) for v in X], [torch.tensor(v) for v in X]), dtype=torch.float32)
            K += 1e-5 * torch.eye(len(X))
            L = torch.cholesky(K)
            alpha = torch.cholesky_solve(y_t.unsqueeze(1), L)
            nll = 0.5 * y_t @ alpha.squeeze() + torch.sum(torch.log(torch.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
            nll.backward()
            optimizer.step()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using a fresh instance of QuantumKernelMethod."""
    kernel = QuantumKernelMethod()
    return kernel.kernel_matrix(a, b)
