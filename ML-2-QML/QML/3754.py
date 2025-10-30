"""Quantum hybrid classifier with random‑layer ansatz and parameterised feature map.

Provides ``build_classifier_circuit`` that returns a
Qiskit circuit, parameter vectors for encoding and variational
layers, and a list of Z‑observables.  The circuit is designed
to be fully differentiable via parameter‑shift or automatic
differentiation frameworks.

The construction incorporates ideas from the regression pair
(e.g. random layers and layered entanglement) while retaining
the simple RX encoding of the original classifier example.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _random_entangling_layer(num_qubits: int) -> QuantumCircuit:
    """
    Generate a fixed random entangling layer composed of
    RZ and CX gates. The pattern is deterministic but
    provides sufficient expressivity for small models.
    """
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.rz(np.random.uniform(0, 2 * np.pi), q)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    qc.cx(num_qubits - 1, 0)
    return qc


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    random_seed: int | None = None,
) -> Tuple[
    QuantumCircuit,
    Iterable[ParameterVector],
    Iterable[ParameterVector],
    List[SparsePauliOp],
]:
    """
    Construct a layered variational circuit with explicit feature
    encoding and trainable weights.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    random_seed : int | None, optional
        Seed for reproducible random entangling pattern.

    Returns
    -------
    circuit : QuantumCircuit
        The full variational ansatz.
    encoding : Iterable[ParameterVector]
        List containing the feature‑encoding parameter vector.
    weights : Iterable[ParameterVector]
        List of parameter vectors for the variational layers.
    observables : list[SparsePauliOp]
        Z‑basis observables for each qubit.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Feature encoding: RX rotations parameterised by input data.
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)

    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    weight_vectors: List[ParameterVector] = []
    for d in range(depth):
        # Single‑qubit rotations
        weight = ParameterVector(f"theta_{d}", num_qubits)
        weight_vectors.append(weight)
        for qubit, w in enumerate(weight):
            circuit.ry(w, qubit)

        # Entangling block
        ent = _random_entangling_layer(num_qubits)
        circuit.compose(ent, inplace=True)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], weight_vectors, observables


__all__ = ["build_classifier_circuit"]
