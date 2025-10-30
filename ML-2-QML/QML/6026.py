"""Quantum kernel with trainable ansatz and training routine."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Parameterized ansatz that encodes input data and has trainable rotation angles."""
    def __init__(self, n_wires: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.params = nn.ParameterList()
        for _ in range(n_layers * n_wires):
            self.params.append(nn.Parameter(torch.randn(1)))
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for w in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=w,
                                 params=x[:, w] if func_name_dict["ry"].num_params else None)
        # Variational layers
        idx = 0
        for _ in range(self.n_layers):
            for w in range(self.n_wires):
                func_name_dict["ry"](q_device, wires=w, params=self.params[idx])
                idx += 1
            # Entangling CNOTs
            for w in range(self.n_wires - 1):
                func_name_dict["cx"](q_device, wires=[w, w + 1])
        # Encode y with negative sign
        for w in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=w,
                                 params=-y[:, w] if func_name_dict["ry"].num_params else None)


class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel with a trainable variational ansatz."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires, n_layers=n_layers)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        mat = np.zeros((len(a), len(b)), dtype=float)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x.unsqueeze(0), y.unsqueeze(0)).item()
        return mat
    def train_parameters(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                         target: np.ndarray, lr: float = 0.01, epochs: int = 200) -> None:
        """Train the ansatz parameters to match a target Gram matrix."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        for epoch in range(epochs):
            optimizer.zero_grad()
            mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
            for i, x in enumerate(a):
                for j, y in enumerate(b):
                    mat[i, j] = self.forward(x.unsqueeze(0), y.unsqueeze(0))
            loss = torch.nn.functional.mse_loss(mat, target_tensor)
            loss.backward()
            optimizer.step()
            if epoch % max(1, (epochs // 10)) == 0:
                print(f"Epoch {epoch}, loss={loss.item():.6f}")


__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
