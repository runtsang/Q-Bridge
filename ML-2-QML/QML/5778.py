"""Quantum‑enhanced kernel and classifier interface.

The :class:`QuantumKernelMethod` class mirrors the classical API but
implements a variational quantum kernel via TorchQuantum and a
data‑uploading variational circuit via Qiskit.  The design keeps the
same public methods so that a downstream model can switch between
classical and quantum back‑ends without code changes.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence, List
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class KernalAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz that mirrors the classical RBF wrapper."""
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


class Kernel(tq.QuantumModule):
    """Quantum kernel module based on a fixed ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
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


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the quantum Gram matrix."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit,
                                                                   Iterable,
                                                                   Iterable,
                                                                   List[SparsePauliOp]]:
    """Construct a variational quantum classifier with data‑encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class QuantumKernelMethod:
    """Unified quantum interface for kernel evaluation and classifier construction."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        self.n_wires = n_wires
        self.depth = depth
        self.kernel = Kernel(n_wires)
        self.q_device = self.kernel.q_device

    def quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def build_classifier(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit,
                                                                     Iterable,
                                                                     Iterable,
                                                                     List[SparsePauliOp]]:
        return build_classifier_circuit(num_qubits, depth)

    def __repr__(self) -> str:
        return f"<QuantumKernelMethod n_wires={self.n_wires} depth={self.depth}>"

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix",
           "build_classifier_circuit", "QuantumKernelMethod"]
