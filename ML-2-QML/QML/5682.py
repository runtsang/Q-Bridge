"""Quantum implementation of the HybridEstimatorQNN architecture.

The circuit encodes a 4‑dimensional feature vector into 4 qubits, applies a trainable random
layer and an RX rotation layer, then measures Pauli‑Z expectations.  The output is a 4‑dimensional
vector that matches the classical model’s dimensionality, enabling head‑to‑head comparisons.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


def _build_circuit() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build a 4‑qubit variational circuit."""
    qc = QuantumCircuit(4)

    # Input parameters
    inp = ParameterVector("x", 4)

    # Encode inputs via Ry rotations
    for i, p in enumerate(inp):
        qc.ry(p, i)

    # Random entangling layer (CNOT chain)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    # Trainable rotation layer
    w = ParameterVector("w", 4)
    for i, p in enumerate(w):
        qc.rx(p, i)

    # Additional entanglement
    qc.cx(0, 2)
    qc.cx(1, 3)

    return qc, inp, w


class HybridEstimatorQNN(QiskitEstimatorQNN):
    """Quantum wrapper inheriting from QiskitEstimatorQNN."""

    def __init__(self) -> None:
        circuit, input_params, weight_params = _build_circuit()
        observables = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])
        estimator = StatevectorEstimator()
        super().__init__(
            circuit=circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )


def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory returning a quantum HybridEstimatorQNN instance."""
    return HybridEstimatorQNN()


__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
