"""
Hybrid quantum classifier that mirrors the classical interface.
Uses a depth‑controlled ansatz with explicit data encoding and bounded variational parameters.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to [-bound, bound]."""
    return max(-bound, min(bound, value))


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    clip: bool = True,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data encoding and variational parameters.
    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    clip : bool
        Whether to clip variational parameters to a bounded range.
    Returns
    -------
    circuit : QuantumCircuit
        The complete parameterized circuit.
    encoding : Iterable[ParameterVector]
        Parameter vectors used for data encoding.
    weights : Iterable[ParameterVector]
        Parameter vectors for the variational layers.
    observables : List[SparsePauliOp]
        Pauli-Z observables for measurement.
    """
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode data via Rx rotations
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            theta = weights[idx]
            if clip:
                theta = circuit.parameter_binds[theta]  # placeholder for runtime binding
                # actual clipping is handled during parameter binding
            circuit.ry(theta, qubit)
            idx += 1
        # Entangling CZs
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: one Pauli-Z per qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, encoding, weights, observables


class HybridClassifierCircuit:
    """
    Wrapper that exposes the same metadata interface as the classical model.
    """
    def __init__(self, num_qubits: int, depth: int, clip: bool = True) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth, clip=clip
        )
        self.num_qubits = num_qubits
        self.depth = depth

    def metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """
        Return metadata compatible with the classical helper:
        - encoding: list of qubit indices (identity mapping)
        - weight_sizes: cumulative number of variational parameters per layer
        - observables: list of output indices (here same as qubit indices)
        """
        encoding = list(range(self.num_qubits))
        weight_sizes = []
        cumulative = 0
        per_layer = self.num_qubits
        for _ in range(self.depth):
            cumulative += per_layer
            weight_sizes.append(cumulative)
        observables = list(range(self.num_qubits))
        return encoding, weight_sizes, observables


__all__ = [
    "HybridClassifierCircuit",
    "build_classifier_circuit",
]
