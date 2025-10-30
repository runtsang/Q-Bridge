"""Extended classical kernel utilities with data preprocessing and hybrid support.

This module builds upon the original RBF kernel by providing a small factory for
classic, quantum, and hybrid kernels.  The classical kernel can be optionally
normalised, the quantum kernel uses a configurable depth of Ry‑only layers,
and the hybrid kernel simply sums the two contributions.  A convenience
``kernel_matrix`` routine accepts NumPy arrays or torch tensors.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence, Union

class BaseKernel(nn.Module):
    """Abstract kernel interface."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ClassicRBF(BaseKernel):
    """Classical radial‑basis‑function kernel with optional normalisation."""
    def __init__(self, gamma: Union[float, torch.Tensor] = 1.0, normalise: bool = False) -> None:
        super().__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32) if not isinstance(gamma, torch.Tensor) else gamma
        self.normalise = normalise

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalise:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
            y = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-8)
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

class QuantumRBF(BaseKernel):
    """Quantum RBF kernel realised with a depth‑controlled Ry ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qdevice = None

    def _build_device(self, batch: int) -> None:
        import torchquantum as tq
        self.qdevice = tq.QuantumDevice(n_wires=self.n_wires, batch_size=batch, device=self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.qdevice is None or self.qdevice.batch_size!= batch:
            self._build_device(batch)
        self.qdevice.reset_states(batch)
        # encode x
        for d in range(self.depth):
            for w in range(self.n_wires):
                self.qdevice.apply_op("ry", wires=[w], params=x[:, w])
        # encode y with negative sign
        for d in range(self.depth):
            for w in range(self.n_wires):
                self.qdevice.apply_op("ry", wires=[w], params=-y[:, w])
        # measurement fidelity (overlap of first qubit with |0>)
        states = self.qdevice.states
        fidelity = torch.abs(states.view(-1, 2**self.n_wires)[:, 0]) ** 2
        return fidelity.view(batch, 1)

class HybridRBF(BaseKernel):
    """Hybrid kernel = classical RBF + quantum RBF."""
    def __init__(self, gamma: Union[float, torch.Tensor] = 1.0, normalise: bool = False,
                 n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.classic = ClassicRBF(gamma, normalise)
        self.quantum = QuantumRBF(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.classic(x, y) + self.quantum(x, y)

def kernel_matrix(a: Sequence[Union[torch.Tensor, np.ndarray]],
                  b: Sequence[Union[torch.Tensor, np.ndarray]],
                  kernel: BaseKernel) -> np.ndarray:
    """Return Gram matrix between two collections of feature vectors."""
    a_t = [torch.as_tensor(ai, dtype=torch.float32) for ai in a]
    b_t = [torch.as_tensor(bi, dtype=torch.float32) for bi in b]
    mat = torch.stack([kernel(x, y) for x in a_t for y in b_t])
    return mat.cpu().numpy().reshape(len(a), len(b))

__all__ = ["BaseKernel", "ClassicRBF", "QuantumRBF", "HybridRBF", "kernel_matrix"]
