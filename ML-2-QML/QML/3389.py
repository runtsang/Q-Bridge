"""Hybrid convolutional filter with quantum circuit and quantum kernel, quantum implementation."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch
from typing import Sequence
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self):
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

class HybridConvKernel:
    """Hybrid filter that emulates a quantum convolution via a parameterised circuit
    and evaluates a quantum RBF‑style kernel using TorchQuantum.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 100,
                 threshold: float = 127,
                 use_kernel: bool = False,
                 gamma: float = 1.0,
                 kernel_data: Sequence[np.ndarray] | None = None):
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.use_kernel = use_kernel
        self.gamma = gamma

        # Quantum convolution circuit
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        # Quantum kernel
        self.kernel = Kernel()
        self.kernel_matrix = None
        if self.use_kernel and kernel_data is not None:
            tensors = [torch.tensor(x, dtype=torch.float32).view(1, -1) for x in kernel_data]
            self.kernel_matrix = np.array([[self.kernel(x, y).item() for y in tensors] for x in tensors])

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum convolution and optionally the quantum kernel."""
        # Quantum convolution
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        conv_out = counts / (self.shots * self.n_qubits)

        if self.use_kernel and self.kernel_matrix is not None:
            vec = torch.tensor(data, dtype=torch.float32).view(1, -1)
            sims = []
            for row in self.kernel_matrix:
                sims.append(np.exp(-self.gamma * np.sum((vec - row)**2)))
            return conv_out + np.mean(sims)
        return conv_out

def Conv() -> HybridConvKernel:
    """Return a drop‑in quantum replacement for the filter."""
    return HybridConvKernel()

__all__ = ["HybridConvKernel", "Conv"]
