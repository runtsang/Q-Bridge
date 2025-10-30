"""Quantum estimator that mirrors the classical 128‑feature network."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN128(EstimatorQNN):
    """
    Variational circuit that accepts 128 classical features and
    returns two expectation values.

    The circuit encodes each feature with an RX gate, then applies
    two layers of Ry rotations followed by CZ entangling gates.
    """

    def __init__(self, depth: int = 2) -> None:
        num_qubits = 128
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)]),
        ]

        estimator = StatevectorEstimator()

        super().__init__(
            circuit=qc,
            observables=observables,
            input_params=[encoding],
            weight_params=[weights],
            estimator=estimator,
        )

def EstimatorQNN() -> EstimatorQNN128:
    """Return a pre‑configured instance of EstimatorQNN128."""
    return EstimatorQNN128()

__all__ = ["EstimatorQNN128", "EstimatorQNN"]
