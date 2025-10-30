"""Hybrid classifier using a quantum feature extractor and classical readout."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a quantum circuit with data encoding and variational ansatz.
    The output signature matches the classical version: (circuit, encoding, weights, observables).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Ansatz layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables for measurement (one Z per qubit)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Quantum EstimatorQNN – mirrors the classical regression network
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> "QuantumEstimatorQNN":
    """
    Return a Qiskit EstimatorQNN instance for regression.
    The construction follows the original EstimatorQNN example, but is
    wrapped in a function to keep the API consistent with the classical
    counterpart.
    """
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
    from qiskit.primitives import StatevectorEstimator as Estimator

    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y", 1)])

    estimator = Estimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["build_classifier_circuit", "EstimatorQNN"]
