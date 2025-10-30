"""Hybrid quantum kernel module with programmable ansatz and classifier construction.

This module extends the original quantum implementation by:
- Allowing a user‑defined list of gates for the ansatz, enabling easy experimentation
  with different encoding strategies.
- Providing a unified ``HybridKernelModel`` class that mirrors the classical
  counterpart for direct comparison.
- Adding a ``build_classifier_circuit`` that returns a Qiskit circuit, an
  encoding parameter vector, a variational parameter vector, and a set of
  observables, matching the classical API.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple, List, Any
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz that encodes classical data via a user‑defined list
    of gate specifications. Each entry contains ``input_idx`` (indices of data
    features used as parameters), ``func`` (gate name) and ``wires`` (target qubits).
    """
    def __init__(self, gate_list: List[dict]) -> None:
        super().__init__()
        self.gate_list = gate_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])

        for info in self.gate_list:
            params = (
                x[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        for info in reversed(self.gate_list):
            params = (
                -y[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class HybridKernelModel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap of two encoded states.
    The underlying ansatz is built from a default 4‑qubit rotation stack
    but can be overridden via ``gate_list``.
    """
    def __init__(self, gate_list: List[dict] | None = None) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        default_gates = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]
        self.ansatz = KernalAnsatz(gate_list or default_gates)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# Backward‑compatibility alias
Kernel = HybridKernelModel

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gate_list: List[dict] | None = None) -> np.ndarray:
    """Evaluate the Gram matrix between two collections of feature vectors."""
    kernel = HybridKernelModel(gate_list)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[Any], Iterable[Any], List[SparsePauliOp]]:
    """
    Construct a variational ansatz with explicit data encoding and tunable depth.

    Returns:
        circuit: ``QuantumCircuit`` ready for execution or simulation.
        encoding: list of named parameters used for feature encoding.
        weights: list of variational parameters.
        observables: Pauli‑Z observables on each qubit for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data‑encoding layer
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers with entanglement
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables

__all__ = ["KernalAnsatz", "HybridKernelModel", "Kernel", "kernel_matrix", "build_classifier_circuit"]
