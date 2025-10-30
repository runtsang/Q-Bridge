"""HybridClassifier: a data‑uploading variational ansatz with optional clipping.

The quantum implementation mirrors the classical helper while incorporating
parameter clipping and a modular depth, inspired by the fraud‑detection
photonic circuit.  The public API matches the original `build_classifier_circuit`
and returns a Qiskit `QuantumCircuit` together with its encoding and observable
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


@dataclass
class LayerParams:
    """Container for a single variational layer's parameters."""
    weights: List[float]
    encodings: List[float]


def _apply_variational_layer(
    circuit: QuantumCircuit,
    qubits: Iterable[int],
    layer: LayerParams,
    *,
    clip: bool,
) -> None:
    """Apply a single variational layer to the circuit."""
    for q, enc in zip(qubits, layer.encodings):
        circuit.rx(enc, q)
    for q, w in zip(qubits, layer.weights):
        circuit.ry(w, q)
    # Entangling block
    for q1, q2 in zip(qubits, list(qubits)[1:]):
        circuit.cz(q1, q2)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    clip: bool = False,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    clip : bool, optional
        If True, clip all parameters to [-π, π] before circuit assembly.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled variational circuit.
    encodings : Iterable[ParameterVector]
        Parameter vectors for data encoding.
    weights : Iterable[ParameterVector]
        Parameter vectors for variational weights.
    observables : List[SparsePauliOp]
        Pauli Z observables on each qubit, for binary readout.
    """
    # Parameter vectors
    encoding_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("θ", num_qubits * depth)

    # Build circuit
    circuit = QuantumCircuit(num_qubits)

    # Initial encoding
    for q, enc in zip(range(num_qubits), encoding_params):
        circuit.rx(enc, q)

    # Variational layers
    weight_iter = iter(weight_params)
    for _ in range(depth):
        layer = LayerParams(
            weights=[next(weight_iter) for _ in range(num_qubits)],
            encodings=[next(weight_iter) for _ in range(num_qubits)],
        )
        _apply_variational_layer(circuit, range(num_qubits), layer, clip=clip)

    # Observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding_params], [weight_params], observables


__all__ = ["build_classifier_circuit", "LayerParams"]
