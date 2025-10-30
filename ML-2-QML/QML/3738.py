"""Hybrid kernel using TorchQuantum with optional shot‑noise simulator."""

from __future__ import annotations

from typing import Sequence, Iterable, List
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridKernel(tq.QuantumModule):
    """Quantum kernel that encodes two inputs via a symmetric ansatz."""
    def __init__(self, n_wires: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.gamma = gamma
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """Return a list of gate specifications for the embedding."""
        return [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum kernel for two 1‑D tensors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        # encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # encode -y (reverse direction)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        return np.array([[self(x, y).item() for y in b] for x in a])

class FastHybridEstimator(tq.QuantumModule):
    """Wrapper that evaluates a quantum kernel with optional shot‑noise."""
    def __init__(self, kernel: HybridKernel, shots: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.kernel = kernel
        self.shots = shots
        self.seed = seed

    def evaluate(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        mat = self.kernel.kernel_matrix(X, Y)
        if self.shots is None:
            return mat
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, 1 / np.sqrt(self.shots), mat.shape)
        return mat + noise

__all__ = ["HybridKernel", "FastHybridEstimator"]
