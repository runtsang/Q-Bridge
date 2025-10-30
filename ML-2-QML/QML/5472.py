"""Quantum circuit for the hybrid classifier.

The module defines a quantum circuit that mirrors the classical
``QuantumHybridClassifier``.  It uses a data‑encoding layer, a stack of
variational layers, and a pooling stage that reduces the width of the
circuit.  The circuit shares its trainable parameters with the
classical side, enabling joint optimisation.

The API is designed to be compatible with the `build_hybrid_classifier`
function defined in the ML module.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_hybrid_classifier(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a hybrid quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits that will encode the input features.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The fully constructed variational circuit.
    encoding : List[Parameter]
        Parameters that encode the classical input.
    weights : List[Parameter]
        Trainable parameters shared with the classical side.
    observables : List[SparsePauliOp]
        Measurement operators for the output.
    """
    # 1. Encoding layer: simple RX rotations on each qubit
    encoding = ParameterVector("x", num_qubits)

    # 2. Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circ = QuantumCircuit(num_qubits)

    # Encode the input
    for qubit in range(num_qubits):
        circ.rx(encoding[qubit], qubit)

    # Variational layers with optional pooling
    weight_idx = 0
    for _ in range(depth):
        # Rotations on all qubits
        for qubit in range(num_qubits):
            circ.ry(weights[weight_idx], qubit)
            weight_idx += 1
        # Entangling CZ between neighbours
        for qubit in range(num_qubits - 1):
            circ.cz(qubit, qubit + 1)

        # Simple pooling: apply a CZ to the last two qubits and
        # measure the last qubit (measurement is optional and can be
        # ignored in post‑processing).  This mimics the QCNN pooling
        # concept without reducing the logical qubit count.
        if num_qubits >= 2:
            circ.cz(num_qubits - 2, num_qubits - 1)
            # Measurement into a classical bit (kept for completeness)
            circ.measure(num_qubits - 1, num_qubits - 1)

    # Observables: measure each remaining qubit in Z basis
    observables = [SparsePauliOp(f"Z{'I'*(num_qubits-1)}")]
    return circ, list(encoding), list(weights), observables
