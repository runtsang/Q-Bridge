"""Hybrid quantum kernel and classifier implementation.

This module mirrors the classical counterpart but replaces the
kernel with a parameter‑driven quantum circuit and the classifier
with a variational quantum circuit.  The interface remains the same,
allowing direct side‑by‑side comparison of classical and quantum
pipelines.

Key features:
- Quantum RBF‑style kernel using TorchQuantum.
- Quantum classifier built with Qiskit, exposing encoding, weights,
  and observables.
- ``HybridKernelClassifier`` encapsulates both components and
  provides ``fit``/``predict`` stubs for future experimentation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class KernalAnsatz(tq.QuantumModule):
    """Quantum ansatz that encodes classical data via a list of gates."""

    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

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
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a quantum variational classifier ansatz."""
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
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class HybridKernelClassifier:
    """Hybrid kernel + classifier for quantum experiments.

    The class stores a quantum kernel and a variational classifier circuit.
    ``fit`` and ``predict`` are placeholders; the actual training logic
    would involve a hybrid variational loop or a classical post‑processing
    step on the kernel features.
    """

    def __init__(self, kernel: Kernel, circuit: QuantumCircuit, epochs: int = 100):
        self.kernel = kernel
        self.circuit = circuit
        self.epochs = epochs
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Placeholder fit method – real implementation would optimize circuit parameters."""
        self.X_train = X
        self.y_train = y
        # A full variational training loop would go here.

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return raw expectation values from the quantum circuit."""
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        # Compute kernel matrix as a placeholder for classification
        K = torch.tensor(kernel_matrix(X, self.X_train), dtype=torch.float32)
        # In a real scenario, we would feed K into a classical post‑processing
        # or use the variational circuit directly on encoded data.
        return torch.argmax(K, dim=1)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel matrix as a torch tensor."""
        return torch.tensor(kernel_matrix(X, Y), dtype=torch.float32)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "build_classifier_circuit",
    "HybridKernelClassifier",
]
