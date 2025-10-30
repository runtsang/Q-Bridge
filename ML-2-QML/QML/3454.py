"""Quantum circuit factory and kernel construction for hybrid experiments.

The module provides:
* ``QuantumClassifierModel.build_classifier_circuit`` – a configurable
  variational ansatz with data‑encoding RX gates, optional RY rotations,
  and a choice of entanglement pattern.
* ``QuantumKernel`` – a TorchQuantum module that evaluates a fixed
  quantum kernel, mirroring the classical RBF kernel in the seed
  module.

Both APIs mirror the classical counterpart, enabling side‑by‑side
benchmarking of classical and quantum classifiers.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# Variational classifier circuit
# --------------------------------------------------------------------------- #

class QuantumClassifierModel:
    """Variational classifier circuit factory.

    The circuit encodes features with RX gates, then applies a stack of
    layers consisting of RY rotations and CZ entanglement.  The depth,
    entanglement scheme, and whether to add a second RY rotation per
    layer are all configurable so the user can explore different
    circuit topologies while keeping the same ``build_classifier_circuit``
    signature as the classical module.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        *,
        entanglement: str = "full",
        use_ry: bool = True,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a Qiskit circuit, encoding parameters, variational
        parameters, and observables.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (features).
        depth : int
            Number of ansatz layers.
        entanglement : {"full", "linear", "none"}, default "full"
            Entanglement pattern inside each layer.
        use_ry : bool, default True
            Whether to apply a second RY rotation after the entanglement
            block.

        Returns
        -------
        Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
            ``circuit``: the constructed Qiskit circuit.
            ``encoding``: list of ParameterVectors for data encoding.
            ``weights``: list of ParameterVectors for variational
            parameters.
            ``observables``: PauliZ observables on each qubit.
        """
        # Data‑encoding rotations
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector(
            "theta",
            num_qubits * depth * (2 if use_ry else 1)
        )

        circuit = QuantumCircuit(num_qubits)

        # Encode each feature with an RX gate
        for qubit in range(num_qubits):
            circuit.rx(encoding[qubit], qubit)

        weight_idx = 0
        for _ in range(depth):
            # First RY rotation per qubit
            for qubit in range(num_qubits):
                circuit.ry(weights[weight_idx], qubit)
                weight_idx += 1

            # Entanglement pattern
            if entanglement == "full":
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        circuit.cz(i, j)
            elif entanglement == "linear":
                for i in range(num_qubits - 1):
                    circuit.cz(i, i + 1)

            # Optional second RY rotation
            if use_ry:
                for qubit in range(num_qubits):
                    circuit.ry(weights[weight_idx], qubit)
                    weight_idx += 1

        # Observables: Pauli‑Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data and applies a fixed sequence of gates."""

    def __init__(self, func_list):
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

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel based on a deterministic TorchQuantum ansatz."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires: int = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumClassifierModel", "QuantumKernel", "kernel_matrix"]
