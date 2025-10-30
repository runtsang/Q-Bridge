from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumKernel(tq.QuantumModule):
    """Variational quantum kernel that learns a similarity function."""
    def __init__(self, n_wires: int, depth: int, param_list: list | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = self._build_ansatz(param_list)

    def _build_ansatz(self, param_list: list | None):
        """Create a list of gate descriptions for the kernel circuit."""
        if param_list is None:
            param_list = []
        # Simple Ry encoding per qubit
        for idx in range(self.n_wires):
            param_list.append({"input_idx": [idx], "func": "ry", "wires": [idx]})
        # Variational layers
        for d in range(self.depth):
            for idx in range(self.n_wires):
                param_list.append({"input_idx": [idx], "func": "ry", "wires": [idx]})
            for idx in range(self.n_wires - 1):
                param_list.append({"input_idx": [], "func": "cz", "wires": [idx, idx + 1]})
        return param_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # encode y (negative phase)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int, depth: int) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    kernel = QuantumKernel(n_wires, depth)
    return np.array([[kernel.forward_kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a Qiskit circuit that mirrors the classical feed‑forward network."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Encoding layer
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        # Variational Ry layer
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling CZ layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["QuantumKernel", "kernel_matrix", "build_classifier_circuit"]
