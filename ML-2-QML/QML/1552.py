"""Quantum kernel construction with a simple variational ansatz."""

from __future__ import annotations

from typing import Sequence, Optional, Iterable

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Variational ansatz encoding classical data via rotation gates
    followed by a trainable layer of rotations and entangling gates.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2,
                 params: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        if params is None:
            params = torch.randn(depth, n_wires, 3, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode x and y into the circuit and apply the variational ansatz.
        x, y: [batch, features] where features == n_wires
        """
        batch = x.shape[0]
        q_device.reset_states(batch)
        # Input encoding: rotate each qubit by the data‑dependent angles
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[i], params=x[:, i].unsqueeze(1))
            func_name_dict["ry"](q_device, wires=[i], params=-y[:, i].unsqueeze(1))
        # Variational layers
        for d in range(self.depth):
            for i in range(self.n_wires):
                angles = self.params[d, i]  # shape: [3]
                func_name_dict["rx"](q_device, wires=[i], params=angles[0].unsqueeze(0))
                func_name_dict["ry"](q_device, wires=[i], params=angles[1].unsqueeze(0))
                func_name_dict["rz"](q_device, wires=[i], params=angles[2].unsqueeze(0))
            # Entangling layer (CNOT chain)
            for i in range(self.n_wires - 1):
                func_name_dict["cx"](q_device, wires=[i, i + 1])

    def overlap(self, q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Return the overlap |<0|ψ>|^2 as a proxy for the kernel value.
        """
        return torch.abs(q_device.states.view(-1)[0]).unsqueeze(0)

class Kernel(tq.QuantumModule):
    """
    Quantum kernel module that evaluates the overlap between two encoded states.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires, depth=depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the kernel matrix for batches x and y.
        x, y: [batch, features] where features == n_wires
        """
        batch_x = x.shape[0]
        batch_y = y.shape[0]
        result = torch.empty((batch_x, batch_y), dtype=torch.float32)
        for i in range(batch_x):
            for j in range(batch_y):
                self.ansatz(self.q_device, x[i:i+1], y[j:j+1])
                result[i, j] = self.ansatz.overlap(self.q_device).item()
        return result

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_wires: int = 4, depth: int = 2) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors using the quantum kernel.
    """
    if not a or not b:
        return np.array([[]], dtype=float)
    kernel = Kernel(n_wires=n_wires, depth=depth)
    a_tensor = torch.stack([t.float() for t in a])  # shape: [len(a), n_wires]
    b_tensor = torch.stack([t.float() for t in b])  # shape: [len(b), n_wires]
    return kernel(a_tensor, b_tensor).detach().cpu().numpy()

def optimize_variational_params(kernel: Kernel,
                                train_x: Sequence[torch.Tensor],
                                train_y: Sequence[torch.Tensor],
                                epochs: int = 100,
                                lr: float = 0.01) -> torch.Tensor:
    """
    Simple gradient‑based optimization of the ansatz parameters
    to maximize similarity for same‑class pairs and minimize for different‑class pairs.
    This is a toy example and may not converge for real data.
    """
    optimizer = torch.optim.Adam(kernel.ansatz.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        mat = kernel(torch.stack([t.float() for t in train_x]),
                     torch.stack([t.float() for t in train_y]))
        loss = -mat.mean()
        loss.backward()
        optimizer.step()
    return kernel.ansatz.params.detach()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "optimize_variational_params"]
