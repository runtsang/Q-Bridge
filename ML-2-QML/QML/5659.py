"""Quantum kernel method with noise‑aware simulation and parameter‑shift gradient."""

from __future__ import annotations

from typing import Sequence, List, Dict, Any

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a quantum circuit with parameter‑shift capability."""
    def __init__(self, func_list: List[Dict[str, Any]]):
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

    def parameter_shift_gradient(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor,
                                 shift: float = np.pi / 2) -> torch.Tensor:
        """Compute gradient of the kernel w.r.t. input x using parameter‑shift rule."""
        grads = []
        for i in range(x.shape[1]):
            x_pos = x.clone()
            x_neg = x.clone()
            x_pos[:, i] += shift
            x_neg[:, i] -= shift
            self.forward(q_device, x_pos, y)
            k_pos = torch.abs(q_device.states.view(-1)[0])
            self.forward(q_device, x_neg, y)
            k_neg = torch.abs(q_device.states.view(-1)[0])
            grads.append((k_pos - k_neg) / (2.0 * shift))
        return torch.stack(grads, dim=-1)

    def apply_noise(self, q_device: tq.QuantumDevice, noise_level: float = 0.01) -> None:
        """Apply amplitude damping noise to each qubit."""
        for wire in range(q_device.n_wires):
            tq.noise.amplitude_damping(q_device, wires=[wire], p=noise_level)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz with noise and gradient support."""
    def __init__(self, n_wires: int = 4, noise_level: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.noise_level = noise_level
        self.ansatz = KernalAnsatz(
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
        if self.noise_level > 0.0:
            self.apply_noise()
        return torch.abs(self.q_device.states.view(-1)[0])

    def apply_noise(self) -> None:
        for wire in range(self.q_device.n_wires):
            tq.noise.amplitude_damping(self.q_device, wires=[wire], p=self.noise_level)

    def parameter_shift_gradient(self, x: torch.Tensor, y: torch.Tensor,
                                 shift: float = np.pi / 2) -> torch.Tensor:
        return self.ansatz.parameter_shift_gradient(self.q_device, x, y, shift)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])


class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel method with noise‑aware simulation and parameter‑shift gradient."""
    def __init__(self, n_wires: int = 4, noise_level: float = 0.0) -> None:
        super().__init__()
        self.kernel = Kernel(n_wires=n_wires, noise_level=noise_level)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.kernel.kernel_matrix(X, Y)

    def parameter_shift_gradient(self, X: torch.Tensor, Y: torch.Tensor,
                                 shift: float = np.pi / 2) -> torch.Tensor:
        return self.kernel.parameter_shift_gradient(X, Y, shift)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], noise_level: float = 0.0) -> np.ndarray:
    kernel = Kernel(noise_level=noise_level)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "QuantumKernelMethod", "kernel_matrix"]
