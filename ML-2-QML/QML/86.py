"""Quantum kernel construction using TorchQuantum with a trainable classical‑to‑quantum mapping."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch.nn as nn

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel evaluated via a parameter‑shift ansatz with a learnable mapping."""
    def __init__(self,
                 n_wires: int = 4,
                 mapping_hidden: int = 8,
                 device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.mapping = nn.Sequential(
            nn.Linear(n_wires, mapping_hidden),
            nn.ReLU(),
            nn.Linear(mapping_hidden, n_wires)
        )
        self.to(device)

    def _apply_ansatz(self, params: torch.Tensor) -> None:
        """Apply the parameter‑shift circuit to the quantum device."""
        for idx, wire in enumerate(range(self.n_wires)):
            func_name_dict["ry"](self.q_device, wires=[wire], params=params[:, idx])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel between two batches."""
        x = x.to(self.q_device.device)
        y = y.to(self.q_device.device)
        # Map classical data to circuit parameters
        params_x = self.mapping(x)
        params_y = self.mapping(y)
        self.q_device.reset_states(x.shape[0])
        # Forward sweep
        self._apply_ansatz(params_x)
        # Reverse sweep with negative parameters
        self._apply_ansatz(-params_y)
        # Return absolute overlap of the first state with the initial basis
        return torch.abs(self.q_device.states.view(-1)[0])

    def train_kernel(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     targets: torch.Tensor,
                     lr: float = 1e-3,
                     epochs: int = 200) -> None:
        """End‑to‑end optimisation of the mapping."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(x, y).squeeze()
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inference wrapper that detaches the result."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    model = QuantumKernelMethod()
    return np.array([[model.forward(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
