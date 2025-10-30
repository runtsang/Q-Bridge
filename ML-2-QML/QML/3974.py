"""Quantum implementation of the classifier with hybrid circuit and kernel."""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """
    Quantum implementation mirroring the classical API.
    - build_classifier_circuit constructs a parameterised ansatz suitable for data‑uploading.
    - build_kernel returns a quantum kernel based on a fixed TorchQuantum ansatz.
    """
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Build a layered Qiskit circuit with data encoding and variational parameters."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circ = QuantumCircuit(num_qubits)

        # Data encoding
        for qubit in range(num_qubits):
            circ.rx(encoding[qubit], qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circ.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circ.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I"*i + "Z" + "I"*(num_qubits-i-1)) for i in range(num_qubits)]
        return circ, [encoding], [weights], observables

    @staticmethod
    def build_kernel() -> "QuantumKernel":
        """Return a quantum kernel instance that emulates the classical RBF kernel API."""
        return QuantumKernel()

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz using a fixed list of Ry gates."""
    def __init__(self, func_list: List[dict]) -> None:
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

class QuantumKernel(tq.QuantumModule):
    """Wraps KernalAnsatz to expose a classical kernel interface."""
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

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumClassifierModel", "QuantumKernel", "kernel_matrix", "KernalAnsatz"]
