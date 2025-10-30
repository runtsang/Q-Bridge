"""Hybrid quantum kernel using TorchQuantum ansatz with batch evaluation and optional Gaussian shot noise."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List

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

class HybridKernel(tq.QuantumModule):
    """Quantum kernel built from a parameterized ansatz, supporting batch evaluation and Gaussian shot noise."""
    def __init__(self, n_wires: int = 4, shots: int | None = None, seed: int | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def evaluate(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor], parameter_sets: Sequence[Sequence[float]]) -> List[np.ndarray]:
        """Return kernel matrices for each parameter set. Parameter sets are ignored for the current ansatz but kept for API compatibility."""
        results: List[np.ndarray] = []
        for _ in parameter_sets:
            K = self.kernel_matrix(X, Y)
            if self.shots is not None:
                noise = self.rng.normal(0, max(1e-6, 1.0 / self.shots), K.shape)
                K += noise
            results.append(K)
        return results

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
