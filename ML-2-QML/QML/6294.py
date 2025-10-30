"""
QuantumKernelMethod (quantum implementation).

Implements a TorchQuantum‑based kernel and a Qiskit parameterized sampler.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler


class KernalAnsatz(tq.QuantumModule):
    """
    Encodes classical data into quantum gates and performs a reverse pass.
    """
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
    """
    Quantum kernel that returns the overlap of two encoded states.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
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


class SamplerQNN:
    """
    Parameterized Qiskit circuit that can be sampled with a statevector sampler.
    """
    def __init__(self, n_wires: int = 2):
        self.n_wires = n_wires
        self.inputs = ParameterVector("input", n_wires)
        self.weights = ParameterVector("weight", 4)
        self.circuit = self._build_circuit()
        self.sampler = Sampler()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_wires)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        return qc

    def sample(self, parameters: dict):
        """
        Execute the sampler with the provided parameter mapping.
        """
        return self.sampler.run(self.circuit, parameters=parameters).result()


class QuantumKernelMethod:
    """
    Public API that exposes the quantum kernel and sampler.
    """
    def __init__(self, n_wires: int = 4):
        self.kernel = Kernel(n_wires)
        self.sampler = SamplerQNN(n_wires // 2)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of 1‑D tensors.
        """
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def sample(self, parameters: dict):
        """
        Run the quantum sampler with the supplied parameter dictionary.
        """
        return self.sampler.sample(parameters)
    
__all__ = ["KernalAnsatz", "Kernel", "SamplerQNN", "QuantumKernelMethod"]
