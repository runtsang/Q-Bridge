import torch
import torchquantum as tq
import numpy as np
from typing import Sequence

class QuantumKernalAnsatz(tq.QuantumModule):
    """Quantum ansatz that encodes two input vectors into a single device."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x, then unencode y with negative angles."""
        self.q_device.reset_states(x.shape[0])
        for idx in range(self.n_wires):
            tq.ry(self.q_device, wires=idx, params=x[:, idx])
        for idx in reversed(range(self.n_wires)):
            tq.ry(self.q_device, wires=idx, params=-y[:, idx])

class QuantumKernel(tq.QuantumModule):
    """Fixed‑parameter quantum kernel based on overlap of encoded states."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.ansatz = QuantumKernalAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(x, y)
        return torch.abs(self.ansatz.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumKernelHybrid(tq.QuantumModule):
    """
    Quantum‑centric hybrid module that couples a fixed quantum kernel
    with a lightweight classical head for supervised learning.
    """
    def __init__(self, n_wires: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.kernel = QuantumKernel(n_wires)
        self.head_linear = torch.nn.Linear(1, hidden_dim)
        self.head_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            k = self.kernel(x, x).unsqueeze(-1).mean(dim=-2)
        else:
            k = self.kernel(x, y).unsqueeze(-1)
        h = torch.relu(self.head_linear(k))
        return self.head_out(h)

__all__ = ["QuantumKernalAnsatz", "QuantumKernel", "kernel_matrix", "QuantumKernelHybrid"]
