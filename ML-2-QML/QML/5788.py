"""Quantum feature extraction and kernel construction for hybrid models."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    observables: Optional[List[SparsePauliOp]] = None,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a data‑re‑uploading ansatz with explicit encoding and
    variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    observables : list[SparsePauliOp] | None
        Observables to measure. If ``None`` a single Z per qubit is used.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised circuit.
    encoding : list[ParameterVector]
        Parameters used for data encoding.
    weights : list[ParameterVector]
        Variational parameters.
    observables : list[SparsePauliOp]
        Measurement operators.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables
    observables = observables or [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], [weights], observables


class QuantumKernel:
    """
    Evaluate a data‑re‑uploading quantum kernel via a fixed circuit.
    """

    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.simulator = AerSimulator(method="statevector")

    def _prepare_circuit(self, x: torch.Tensor, y: torch.Tensor) -> QuantumCircuit:
        """
        Encode two data points with opposite signs to obtain the overlap kernel.
        """
        qc = self.circuit.copy()
        params = {
            self.encoding[0]: x.numpy().flatten(),
            self.weights[0]: np.zeros_like(self.weights[0].numpy()),
        }
        # Forward data
        qc.bind_parameters(params)
        # Reverse encoding for y
        rev_params = {
            self.encoding[0]: -y.numpy().flatten(),
            self.weights[0]: np.zeros_like(self.weights[0].numpy()),
        }
        qc.bind_parameters(rev_params, inplace=True)
        return qc

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value k(x, y) = |⟨x|y⟩|^2.
        """
        qc = self._prepare_circuit(x, y)
        result = self.simulator.run(qc).result()
        state = result.get_statevector(qc)
        return torch.abs(state[0]) ** 2

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Evaluate the Gram matrix between two datasets.
        """
        kernel = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                kernel[i, j] = self(x, y).item()
        return kernel


__all__ = ["build_classifier_circuit", "QuantumKernel"]
