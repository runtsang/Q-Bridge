"""HybridSamplerQNN: quantum sampler network with quantum kernel.

This module builds on the original SamplerQNN by embedding a parameterized
quantum circuit (via TorchQuantum) and exposing a quantum RBF‑style kernel
through a fixed ansatz.  The class can be used as a drop‑in replacement
for the classical version while leveraging quantum state overlaps for
similarity estimation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that maps two classical inputs onto a single device."""
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

class Kernel(tq.QuantumModule):
    """Quantum kernel module that evaluates the overlap of two encoded states."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
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
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridSamplerQNN(tq.QuantumModule):
    """Quantum sampler network that also exposes a quantum kernel."""
    def __init__(self, use_kernel: bool = False) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Weight parameters for the parameterized circuit
        self.weights = tq.ParameterVector("weight", 4)

        if self.use_kernel:
            self.kernel = Kernel()
        else:
            self.kernel = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = inputs.shape[0]
        self.q_device.reset_states(batch)

        # Encode inputs
        self.q_device.ry(inputs[:, 0], 0)
        self.q_device.ry(inputs[:, 1], 1)
        self.q_device.cx(0, 1)

        # Parameterized part
        self.q_device.ry(self.weights[0], 0)
        self.q_device.ry(self.weights[1], 1)
        self.q_device.cx(0, 1)
        self.q_device.ry(self.weights[2], 0)
        self.q_device.ry(self.weights[3], 1)

        probs = self.q_device.states.view(-1).real
        if self.kernel is not None:
            return probs, self.kernel(probs.unsqueeze(1), probs.unsqueeze(1))
        return probs

def SamplerQNN(use_kernel: bool = False) -> HybridSamplerQNN:
    """Factory returning a HybridSamplerQNN instance."""
    return HybridSamplerQNN(use_kernel=use_kernel)

__all__ = ["SamplerQNN", "HybridSamplerQNN", "Kernel", "kernel_matrix"]
