import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List, Optional

class RBFKernel(nn.Module):
    """Classical radial basis function kernel implemented in PyTorch."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumAnsatz(tq.QuantumModule):
    """Simple quantum ansatz that applies Ry rotations and reverses them."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(nn.Module):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = QuantumAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridKernel:
    """
    Unified interface that can compute either a classical RBF kernel or a
    quantum kernel via TorchQuantum.  It also provides a fast evaluation
    routine with optional Gaussian shot noise, mirroring FastBaseEstimator.
    """
    def __init__(self, gamma: float = 1.0, use_quantum: bool = False, n_wires: int = 4):
        self.gamma = gamma
        self.use_quantum = use_quantum
        if use_quantum:
            self.kernel = QuantumKernel(n_wires=n_wires)
        else:
            self.kernel = RBFKernel(gamma)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two batches of tensors."""
        mat = np.empty((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.kernel(x, y).item()
        return mat

    def evaluate(self, X: Sequence[Sequence[float]], Y: Sequence[Sequence[float]],
                 *, shots: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """
        Compute the kernel matrix for two collections of data points.
        If ``shots`` is provided, Gaussian noise with variance 1/shots is added.
        """
        X_t = [torch.tensor(x, dtype=torch.float32) for x in X]
        Y_t = [torch.tensor(y, dtype=torch.float32) for y in Y]
        K = self.kernel_matrix(X_t, Y_t)
        if shots is None:
            return K
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0 / np.sqrt(shots), size=K.shape)
        return K + noise

__all__ = ["HybridKernel"]
