"""Hybrid kernel classifier – quantum implementation.

The quantum side implements a variational RBF‑style kernel with
TorchQuantum and a Qiskit classifier circuit.  The public API matches
the classical implementation so the two modules can be swapped
seamlessly.  The module exposes :class:`HybridKernelClassifier`
and a helper :func:`build_classifier_circuit`.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumRBFKernel(tq.QuantumModule):
    """Quantum RBF‑style kernel based on a fixed variational ansatz."""

    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class HybridKernelClassifier(tq.QuantumModule):
    """
    Quantum hybrid kernel classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int, default 2
        Depth of the variational ansatz used both for the kernel and the classifier.
    """

    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Kernel ansatz
        self.kernel_ansatz = QuantumRBFKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Classifier circuit
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum kernel between batches `x` and `y`.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.kernel_ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.
        """
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def build_classifier_circuit(
        self, num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a simple layered ansatz with explicit encoding and variational parameters.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.

    Returns
    -------
    circuit : QuantumCircuit
        Variational circuit with data‑encoding and entangling layers.
    encoding : Iterable
        ParameterVector for data encoding.
    weights : Iterable
        ParameterVector for variational parameters.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridKernelClassifier", "build_classifier_circuit"]
